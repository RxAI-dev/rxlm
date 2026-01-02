import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import math
from typing import Union, Optional
from datetime import datetime
from sklearn.metrics import f1_score
from ..training.base import BaseTrainer
from .models import JointTrainingModel
from .ddp import distributed_mean
from ..compile import compile_training_model, CompileConfig, is_compile_available


class JointLMTrainer(BaseTrainer):
    """
    Joint LM Trainer is made for decoder and encoder training on MLM and autoregressive objectives. Training
    includes memory cross-attention, that works like in original encoder-decoder transformer in this stage.

    It's recommended for pre-training and fine-tuning Reactive Transformer components
    """

    def __init__(
            self,
            model: JointTrainingModel,
            device: torch.device,
            vocab_size: int,
            use_amp: bool = False,
            dtype: torch.dtype = None,
            components_loss_log_interval: int = None,
            encoder_loss_scale: float = 1.0,
            decoder_loss_scale: float = 1.0,
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            is_sft: bool = False,
            use_torch_compile: bool = False,
            compile_config: Optional[CompileConfig] = None,
            **kwargs
    ):
        super(JointLMTrainer, self).__init__(model, device, use_amp=use_amp, dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.components_loss_log_interval = components_loss_log_interval
        self.encoder_loss_scale = encoder_loss_scale
        self.decoder_loss_scale = decoder_loss_scale
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale
        self.is_sft = is_sft

        self.use_torch_compile = use_torch_compile
        self.compile_config = compile_config

        # Apply torch.compile if requested
        if self.use_torch_compile and is_compile_available():
            print("Applying torch.compile optimizations to model...")
            self.model = compile_training_model(self.model, self.compile_config)
            print("torch.compile optimizations applied successfully.")
        elif self.use_torch_compile and not is_compile_available():
            print("Warning: torch.compile requested but not available (requires PyTorch 2.0+)")

    def train_step(self, batch: dict[str, Union[torch.Tensor, dict[torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        batch = {
            k: (
                {kk: vv.to(self.device) for kk, vv in v.items()} if not torch.is_tensor(v) else v.to(self.device)
            ) for k, v in batch.items()
        }

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                (encoder_loss, decoder_loss), _ = self.compute_loss(batch)
        elif self.use_te_fp8 and self.fp8_recipe:
            import transformer_engine.pytorch as te
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                (encoder_loss, decoder_loss), _ = self.compute_loss(batch)
        else:
            (encoder_loss, decoder_loss), _ = self.compute_loss(batch)

        if self.components_loss_log_interval is not None:
            if batch_idx % self.components_loss_log_interval == 0:
                print(f"Encoder loss: {encoder_loss.item():.4f}")
                print(f"Decoder loss: {decoder_loss.item():.4f}")
                if self.encoder_loss_scale != 1.0:
                    print(
                        f"Encoder loss scaled by {self.encoder_loss_scale}: {(encoder_loss * self.encoder_loss_scale).item():.4f}")
                if self.decoder_loss_scale != 1.0:
                    print(
                        f"Decoder loss scaled by {self.decoder_loss_scale}: {(decoder_loss * self.decoder_loss_scale).item():.4f}")

        if self.writer is not None and self.total_steps % self.tensorboard_interval == 0:
            self.writer.add_scalar('Encoder loss/Train', encoder_loss.item(), self.total_steps)
            self.writer.add_scalar('Decoder loss/Train', decoder_loss.item(), self.total_steps)
            self.writer.add_scalar('Encoder perplexity/Train', torch.exp(encoder_loss).item(), self.total_steps)
            self.writer.add_scalar('Decoder perplexity/Train', torch.exp(decoder_loss).item(), self.total_steps)

        return (encoder_loss * self.encoder_loss_scale) + (decoder_loss * self.decoder_loss_scale)

    def valid_step(self, batch: dict[str, Union[torch.Tensor, dict[torch.Tensor]]]) -> tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        batch = {
            k: (
                {kk: vv.to(self.device) for kk, vv in v.items()} if not torch.is_tensor(v) else v.to(self.device)
            ) for k, v in batch.items()
        }
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                (encoder_loss, decoder_loss), (encoder_logits, decoder_logits) = self.compute_loss(batch)
        elif self.use_te_fp8 and self.fp8_recipe:
            import transformer_engine.pytorch as te
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                (encoder_loss, decoder_loss), (encoder_logits, decoder_logits) = self.compute_loss(batch)
        else:
            (encoder_loss, decoder_loss), (encoder_logits, decoder_logits) = self.compute_loss(batch)

        return (encoder_loss, decoder_loss), (encoder_logits, decoder_logits)

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        model = next(self.model.children()) if isinstance(self.model, DistributedDataParallel) else self.model

        router_loss = model.decoder.model.moe_router_loss()
        loss = main_loss + self.moe_aux_loss_scale * router_loss

        if self.writer is not None:
            if self.model.training:
                if self.total_steps % self.tensorboard_interval == 0:
                    self.writer.add_scalar('Router aux loss/Train', router_loss.item(), self.total_steps)
                    self.writer.add_scalar('Model loss/Train', main_loss.item(), self.total_steps)
            else:
                self.writer.add_scalar('Router aux loss/Valid', router_loss.item(), self.total_steps)
                self.writer.add_scalar('Model loss/Valid', main_loss.item(), self.total_steps)

        if self.components_loss_log_interval is not None:
            if self.total_steps % self.components_loss_log_interval == 0:
                print(f"Router loss: {router_loss.item():.4f}")

        return loss

    def compute_loss(self, batch: dict[str, dict[str, torch.Tensor]]) -> tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        encoder_inputs = batch['encoder']['input_ids']
        encoder_labels = batch['encoder']['labels']
        decoder_inputs = batch['decoder']['input_ids']
        decoder_targets = batch['decoder']['targets']
        attention_mask = batch['attention_mask']

        encoder_logits, decoder_logits = self.model(
            encoder_inputs,
            decoder_inputs,
            attention_mask=attention_mask,
        )

        encoder_loss = F.cross_entropy(
            encoder_logits.view(-1, self.vocab_size),
            encoder_labels.view(-1),
            ignore_index=-100
        )

        shifted_logits = decoder_logits[:, :-1].contiguous()
        shifted_targets = decoder_targets[:, 1:].contiguous()

        if self.is_sft:
            decoder_loss = F.cross_entropy(
                shifted_logits.view(-1, self.vocab_size),
                shifted_targets.view(-1),
                ignore_index=-100
            )
        else:
            decoder_loss = F.cross_entropy(
                shifted_logits.view(-1, self.vocab_size),
                shifted_targets.view(-1)
            )

        decoder_loss = self._moe_aux_loss(decoder_loss)

        return (encoder_loss, decoder_loss), (encoder_logits, decoder_logits)

    def _valid_writer(self, epoch: int, val_loss: float, val_metrics: dict):
        self.writer.add_scalar('Loss/Valid', val_loss, epoch)
        self.writer.add_scalar('Perplexity/Valid', math.exp(val_loss), epoch)
        if val_metrics['accuracy']:
            self.writer.add_scalar('Encoder node accuracy/Valid', val_metrics['accuracy']['node_encoder'], epoch)
            self.writer.add_scalar('Decoder node accuracy/Valid', val_metrics['accuracy']['node_decoder'], epoch)
            self.writer.add_scalar('Encoder avg. accuracy/Valid', val_metrics['accuracy']['encoder'], epoch)
            self.writer.add_scalar('Decoder avg. accuracy/Valid', val_metrics['accuracy']['decoder'], epoch)
        if val_metrics['loss']:
            self.writer.add_scalar('Encoder loss/Valid', val_metrics['loss']['encoder'], epoch)
            self.writer.add_scalar('Encoder perplexity/Valid', math.exp(val_metrics['loss']['encoder']), epoch)
            self.writer.add_scalar('Decoder accuracy/Valid', val_metrics['loss']['decoder'], epoch)
            self.writer.add_scalar('Decoder perplexity/Valid', math.exp(val_metrics['loss']['decoder']), epoch)

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        val_loss = torch.tensor(0.0).to(self.device)
        dec_loss = torch.tensor(0.0).to(self.device)
        enc_loss = torch.tensor(0.0).to(self.device)
        correct_mlm = torch.tensor(0).to(self.device)
        total_mlm = torch.tensor(0).to(self.device)
        correct_alm = torch.tensor(0).to(self.device)
        total_alm = torch.tensor(0).to(self.device)

        val_dataloader = self._valid_loader(batch_size)

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    (encoder_loss, decoder_loss), (encoder_logits, decoder_logits) = self.valid_step(batch)
                    enc_loss += encoder_loss
                    dec_loss += decoder_loss
                    val_loss += (encoder_loss * self.encoder_loss_scale) + (decoder_loss * self.decoder_loss_scale)

                    encoder_labels = batch['encoder']['labels'].to(self.device)
                    valid_mlm_indices = encoder_labels != -100
                    if valid_mlm_indices.any():
                        preds_mlm = encoder_logits.argmax(-1)
                        correct_mlm += (preds_mlm[valid_mlm_indices] == encoder_labels[valid_mlm_indices]).sum()
                        total_mlm += valid_mlm_indices.sum()

                    shifted_logits = decoder_logits[:, :-1].contiguous()
                    shifted_targets = batch['decoder']['targets'][:, 1:].to(self.device).contiguous()
                    valid_alm_indices = shifted_targets != -100
                    if valid_alm_indices.any():
                        preds_alm = shifted_logits.argmax(-1)
                        correct_alm += (preds_alm[valid_alm_indices] == shifted_targets[valid_alm_indices]).sum()
                        total_alm += valid_alm_indices.sum()

        loader_len = len(val_dataloader)
        avg_loss = val_loss / loader_len
        avg_dec_loss = dec_loss / loader_len
        avg_enc_loss = enc_loss / loader_len
        mlm_acc = (correct_mlm / total_mlm * 100) if total_mlm > 0 else torch.tensor(0.0).to(self.device)
        alm_acc = (correct_alm / total_alm * 100) if total_alm > 0 else torch.tensor(0.0).to(self.device)
        node_mlm_acc = mlm_acc.item()
        node_alm_acc = alm_acc.item()
        if self.use_ddp:
            avg_dec_loss = distributed_mean(avg_dec_loss)
            avg_enc_loss = distributed_mean(avg_enc_loss)
            mlm_acc = distributed_mean(mlm_acc)
            alm_acc = distributed_mean(alm_acc)

        metrics = {
            'accuracy': {
                'encoder': mlm_acc.item(),
                'decoder': alm_acc.item(),
                'node_encoder': node_mlm_acc,
                'node_decoder': node_alm_acc,
            },
            'loss': {
                'encoder': avg_enc_loss.item(),
                'decoder': avg_dec_loss.item(),
            }
        }
        self.model.train()
        return avg_loss, metrics


class IterativeJointLMTrainer(JointLMTrainer):
    """
    JointLMTrainer with batched collection and training for streaming datasets.

    This trainer improves efficiency on large streaming datasets by:
    1. Collecting N batches (with tokenization happening during collection)
    2. Running training loop on the collected batches

    This avoids CPU-bound bottlenecks from loading and tokenizing each batch
    separately during the training loop.
    """
    def __init__(
            self,
            model: JointTrainingModel,
            device: torch.device,
            vocab_size: int,
            collect_n_batches: int = 1000,
            use_amp: bool = False,
            dtype: torch.dtype = None,
            components_loss_log_interval: int = None,
            encoder_loss_scale: float = 1.0,
            decoder_loss_scale: float = 1.0,
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            is_sft: bool = False,
            collect_log_interval: int = 100,
            use_iterable_dataset: bool = True,
            debug_timing: bool = False,
            use_torch_compile: bool = False,
            compile_config: Optional[CompileConfig] = None,
            **kwargs
    ):
        super().__init__(
            model=model,
            device=device,
            vocab_size=vocab_size,
            use_amp=use_amp,
            dtype=dtype,
            components_loss_log_interval=components_loss_log_interval,
            encoder_loss_scale=encoder_loss_scale,
            decoder_loss_scale=decoder_loss_scale,
            use_moe_aux_loss=use_moe_aux_loss,
            moe_aux_loss_scale=moe_aux_loss_scale,
            is_sft=is_sft,
            use_iterable_dataset=use_iterable_dataset,
            use_torch_compile=use_torch_compile,
            compile_config=compile_config,
            **kwargs
        )
        self.collect_n_batches = collect_n_batches
        self.collect_log_interval = collect_log_interval
        self.debug_timing = debug_timing

    def _run_epoch(
            self,
            dataloader: torch.utils.data.DataLoader,
            epoch: int,
            optimizer: torch.optim.Optimizer,
            batch_size: int,
            scaler: torch.cuda.amp.GradScaler = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(self.model, epoch)

        self.accumulated_loss = torch.tensor(0.0, device=self.device)
        self.optimizer_step_count = 0

        accumulated_tokens = torch.tensor(0, device=self.device, dtype=torch.long)

        # Temporary list for collected batches
        collected_batches = []
        base_batch_idx = 0
        collect_idx = 1

        for batch_idx, batch in enumerate(dataloader):
            if not self.is_running:
                break

            # Collect phase: add batch to temporary list (tokenization happens here)
            collected_batches.append(batch)
            collect_idx = collect_idx + 1
            if collect_idx % self.collect_log_interval == 0:
                print(f'Collect & tokenize batch: {collect_idx} / {self.collect_n_batches}')

            # When we've collected N batches, run training loop
            if len(collected_batches) >= self.collect_n_batches:
                if self.use_ddp:
                    torch.distributed.barrier()
                print(f'Train on collected: {self.collect_n_batches} batches')
                self._train_on_collected_batches(
                    collected_batches,
                    optimizer,
                    scheduler,
                    accumulated_tokens,
                    base_batch_idx,
                    batch_size,
                    scaler=scaler,
                )
                base_batch_idx = batch_idx + 1
                # Clear collected batches
                collected_batches = []
                collect_idx = 1

                if self.use_ddp:
                    torch.distributed.barrier()

                if not self.is_running:
                    break

        # Train on any remaining collected batches
        if collected_batches and self.is_running:
            if self.use_ddp:
                torch.distributed.barrier()
            self._train_on_collected_batches(
                collected_batches,
                optimizer,
                scheduler,
                accumulated_tokens,
                base_batch_idx,
                batch_size,
                scaler=scaler,
            )

            if self.use_ddp:
                torch.distributed.barrier()

        # Validation at the end of epoch
        if self.validation_dataset:
            self.validation_steps = 0
            val_loss, val_metrics = self.validate(batch_size)
            if self.use_ddp:
                from .ddp import distributed_value_mean
                val_loss = distributed_value_mean(val_loss, device=self.device)

            self.validation_metrics[epoch] = val_metrics

            if self.writer:
                self._valid_writer(epoch, val_loss, val_metrics)

            for callback in self.callbacks:
                should_stop = callback.on_validation_end(self.model, epoch, val_loss, val_metrics)
                if should_stop:
                    self.is_running = False

        for callback in self.callbacks:
            should_stop = callback.on_epoch_end(self.model, epoch)
            if should_stop:
                self.is_running = False

        if self.writer:
            self.writer.flush()

    def _train_on_collected_batches(
            self,
            collected_batches: list,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            accumulated_tokens: torch.Tensor,
            base_batch_idx: int,
            batch_size: int,
            scaler: torch.cuda.amp.GradScaler = None
    ):
        start_time = None

        pre_forward_time = 0
        forward_time = 0

        forward_times = []
        backward_times = []

        """Train on collected batches"""
        for i, batch in enumerate(collected_batches):
            if not self.is_running:
                break
            if self.get_batch_size(batch) == batch_size:
                if self.debug_timing:
                    if i == 100:
                        start_time = datetime.timestamp(datetime.now())
                    elif i == 200:
                        end_time = datetime.timestamp(datetime.now())
                        b100_time = end_time - start_time
                        b_time = b100_time / 100.0
                        i_time = b_time / batch_size
                        print(f'100 batches time: {b100_time}')
                        print(f'1 batch time: {b_time}')
                        print(f'item time: {i_time}')

                        forward_b_time = sum(forward_times) / len(forward_times)
                        backward_b_time = sum(backward_times) / len(backward_times)

                        print(f'Forward batch: {forward_b_time} / example: {forward_b_time / batch_size}')
                        print(f'Backward batch: {backward_b_time} / example: {backward_b_time / batch_size}')


                if self.debug_timing and 100 < i < 200:
                    pre_forward_time = datetime.timestamp(datetime.now())

                self.total_steps += 1
                self.epoch_steps = base_batch_idx + i + 1
                accumulated_tokens += batch['attention_mask'].sum()

                loss = self.train_step(batch, self.epoch_steps)
                self.accumulated_loss += loss
                loss = loss / self.gradient_accumulation_steps

                if self.debug_timing and 100 < i < 200:
                    forward_time = datetime.timestamp(datetime.now())
                    forward_times.append(forward_time - pre_forward_time)

                if self.use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.debug_timing and 100 < i < 200:
                    backward_times.append(datetime.timestamp(datetime.now()) - forward_time)

                self.optimizer_step_count += 1
                if self.optimizer_step_count % self.gradient_accumulation_steps == 0:
                    # Clip gradients after accumulation
                    if self.use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                    if self.use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    self.accumulated_loss = torch.tensor(0.0, device=self.device)
                    self.optimizer_step_count = 0

                if self.writer and self.total_steps % self.tensorboard_interval == 0:
                    loss_item = (loss * self.gradient_accumulation_steps).item()
                    self.writer.add_scalar(
                        'Loss/train',
                        loss_item,
                        self.total_steps,
                    )
                    self.writer.add_scalar(
                        'Loss/train last epoch',
                        loss_item,
                        self.epoch_steps
                    )
                    self.writer.add_scalar(
                        'Perplexity/train',
                        torch.exp(torch.tensor(loss_item)),
                        self.total_steps,
                    )

                    self.total_tokens += accumulated_tokens.item()
                    accumulated_tokens = torch.tensor(0, device=self.device, dtype=torch.long)
                    self.writer.add_scalar(
                        'Processed tokens',
                        self.total_tokens,
                        self.total_steps
                    )

                for callback in self.callbacks:
                    should_stop = callback.on_batch_end(self.model, self.epoch_steps, loss * self.gradient_accumulation_steps, batch)
                    if should_stop:
                        self.is_running = False


        for callback in self.callbacks:
            should_stop = callback.on_iteration_end(self.model, self.epoch_steps)
            if should_stop:
                self.is_running = False