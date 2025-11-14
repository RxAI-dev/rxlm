import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import math, random
from typing import Union, Callable, Any
from ..training.base import BaseTrainer
from .models import SupervisedMemoryAwareModel, MemoryAttentionTrainingModel
from .ddp import distributed_value_mean, distributed_mean
from .utils import TokenizedDict, smart_concat
from .dataset import MrlCurriculumDataset


class SupervisedMemoryAttentionTrainer(BaseTrainer):
    """
    Supervised Memory Aware Trainer - made to train decoder memory cross-attention to use correct accumulated memory
    states, after basic self-supervised pre-training of memory attention network
    """

    def __init__(
            self,
            model: MemoryAttentionTrainingModel,
            device: torch.device,
            label_weights: Union[list[tuple[float, float]], Callable[[int], tuple[float, float]]],
            max_seq_len: int = 256,
            pad_token_id: int = 0,
            noise_levels: tuple[float, float] = (0.0, 0.0),
            label_weights_random_factor: float = None,
            stm_diff_factor: float = None,
            dataset_collate_fn: Callable[[list[Any]], dict[str, Any]] = MrlCurriculumDataset.collate_mrl_batch,
            **kwargs
    ):
        super(SupervisedMemoryAttentionTrainer, self).__init__(
            model, device, dataset_collate_fn=dataset_collate_fn, **kwargs
        )
        self.get_batch_size = lambda batch: batch['query']['attention_mask'].size(0)
        self.total_inner_steps = 0
        self.valid_inner_steps = 0
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.label_weights = label_weights
        self.noise_levels = noise_levels
        self.label_weights_random_factor = label_weights_random_factor
        self.stm_diff_factor = stm_diff_factor

    def _get_model(self):
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = next(model.children())
        return model

    def reset_stm(self):
        self._get_model().reset_memory()

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

        accumulated_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
        for batch_idx, batch in enumerate(dataloader):
            if not self.is_running:
                break
            else:
                if self.get_batch_size(batch) == batch_size:
                    self.total_steps += 1
                    self.epoch_steps = batch_idx

                    self.reset_stm()

                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']

                    number_of_inner_steps = len(interactions) + 1

                    for inner_step_idx in range(number_of_inner_steps):
                        self.total_inner_steps += 1
                        if inner_step_idx == 0:
                            next_query, next_answer = self._move_multiple_batches(first_query, first_answer)
                        else:
                            next_query, next_answer = self._move_multiple_batches(
                                interactions[inner_step_idx - 1]['query'], interactions[inner_step_idx - 1]['answer']
                            )

                        self._get_model().clone_reset_memory()
                        accumulated_stm = self._get_model().get_memory_state()

                        label_weights = self.label_weights(inner_step_idx) if callable(self.label_weights) else \
                            self.label_weights[inner_step_idx]

                        # Randomize label weights, when factor is set
                        if self.label_weights_random_factor is not None:
                            weights_modifier = random.random() * self.label_weights_random_factor
                            if random.random() > 0.5:
                                label_weights = label_weights[0] + weights_modifier, label_weights[1] - weights_modifier
                            else:
                                label_weights = label_weights[0] - weights_modifier, label_weights[1] + weights_modifier

                        if self.use_amp:
                            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                                train_batch = {
                                    **smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                                   pad_token_id=self.pad_token_id),
                                    'acc_stm': accumulated_stm,
                                }
                                accumulated_tokens += train_batch['attention_mask'].sum()
                                loss, cosine_sim = self.compute_loss(train_batch, weights=label_weights,
                                                                     inner_step_idx=inner_step_idx)
                        else:
                            train_batch = {
                                **smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                               pad_token_id=self.pad_token_id),
                                'acc_stm': accumulated_stm,
                            }
                            accumulated_tokens += train_batch['attention_mask'].sum()
                            loss, cosine_sim = self.compute_loss(train_batch, weights=label_weights,
                                                                 inner_step_idx=inner_step_idx)

                        self.accumulated_loss += loss
                        loss = loss / self.gradient_accumulation_steps

                        if self.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        self.optimizer_step_count += 1
                        if self.optimizer_step_count % self.gradient_accumulation_steps == 0:
                            # Clip gradients after accumulation
                            if self.use_amp:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self._get_model().trainable_parameters(), max_norm=1.0,
                                                           error_if_nonfinite=False)
                            if self.use_amp:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()

                            optimizer.zero_grad()

                            if scheduler is not None:
                                scheduler.step()

                            if self.writer and self.total_inner_steps % self.tensorboard_interval == 0:
                                loss_item = self.accumulated_loss / self.gradient_accumulation_steps
                                self._train_writer(
                                    loss_item, cosine_sim.item(),
                                    epoch_step=(batch_idx * number_of_inner_steps) + inner_step_idx,
                                    inner_step=inner_step_idx,
                                )
                                self.total_tokens += accumulated_tokens.item()
                                accumulated_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
                                self.writer.add_scalar(
                                    'Processed tokens',
                                    self.total_tokens,
                                    self.total_inner_steps
                                )

                            self.accumulated_loss = 0.0
                            self.optimizer_step_count = 0


                        for callback in self.callbacks:
                            should_stop = callback.on_batch_end(
                                self.model, (batch_idx * number_of_inner_steps) + inner_step_idx,
                                loss, train_batch
                            )
                            if should_stop:
                                self.is_running = False

        if self.validation_dataset:
            self.validation_steps = 0
            self.valid_inner_steps = 0
            val_loss, val_metrics = self.validate(batch_size)
            if self.use_ddp:
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

    def _move_batch(self, batch: TokenizedDict) -> TokenizedDict:
        if self.use_amp:
            return {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
            }
        else:
            return {
                'input_ids': batch['input_ids'].to(self.device, dtype=self.dtype),
                'attention_mask': batch['attention_mask'].to(self.device, dtype=self.dtype),
            }

    def _move_multiple_batches(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._move_batch(batch) for batch in batches]

    def compute_loss(
            self,
            batch: dict[str, torch.Tensor],
            weights: tuple[float, float] = (0.5, 0.5),
            inner_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_inputs = batch['input_ids']
        encoder_mask = batch['attention_mask']
        accumulated_stm = batch['acc_stm']

        input_noise, label_noise = self.noise_levels

        new_stm_state, encoded_layers_data = self.model(
            encoder_inputs,
            attention_mask=encoder_mask,
            noise_level=input_noise,
        )

        acc_weight, new_weight = weights
        labels = (acc_weight * accumulated_stm + new_weight * encoded_layers_data) + label_noise * torch.randn_like(
            encoded_layers_data
        )

        cosine_sim = F.cosine_similarity(
            new_stm_state,
            labels,
            dim=-1,
        )

        loss = -cosine_sim.mean()

        if self.stm_diff_factor is not None:
            updated_stm = self._get_model().get_memory_state()
            step_diff = F.mse_loss(updated_stm, accumulated_stm, reduction='mean')
            step_diff_factor = self.stm_diff_factor * torch.sqrt(torch.tensor(inner_step_idx + 1).to(self.device))
            loss += step_diff_factor * step_diff

        return loss, cosine_sim.mean()

    def _train_writer(self, loss: float, cosine_sim: float, epoch_step: int, inner_step: int) -> None:
        self.writer.add_scalar(
            'Loss/train',
            loss,
            self.total_inner_steps,
        )
        self.writer.add_scalar(
            'Loss/train last epoch',
            loss,
            epoch_step,
        )
        self.writer.add_scalar(
            'Cosine sim/train',
            cosine_sim,
            self.total_inner_steps,
        )

        self.writer.add_scalar(
            f'Loss/train (step {inner_step})',
            loss,
            self.total_inner_steps,
        )
        self.writer.add_scalar(
            f'Cosine sim/train (step {inner_step})',
            cosine_sim,
            self.total_inner_steps,
        )

    def _valid_writer(self, epoch: int, val_loss: float, val_metrics: dict):
        self.writer.add_scalar('Loss/Valid', val_loss, epoch)
        if val_metrics['cosine_sim']:
            self.writer.add_scalar('Cosine sim/Valid', val_metrics['cosine_sim'], epoch)
        if val_metrics['rmse_diff']:
            self.writer.add_scalar('RMSE STM Diff/Valid', val_metrics['rmse_diff'], epoch)

    def _valid_step_writer(self, step: int, val_loss: float, val_metrics: dict, inner_step: int):
        self.writer.add_scalar('Loss/Valid step', val_loss, step)
        self.writer.add_scalar(f'Loss/Valid step (step: {inner_step})', val_loss, step)
        if val_metrics['cosine_sim']:
            self.writer.add_scalar('Cosine sim/Valid step', val_metrics['cosine_sim'], step)
            self.writer.add_scalar(f'Cosine sim/Valid step (step: {inner_step})', val_metrics['cosine_sim'], step)
        if val_metrics['rmse_diff']:
            self.writer.add_scalar('RMSE STM Diff/Valid step', val_metrics['rmse_diff'], step)
            self.writer.add_scalar(f'RMSE STM Diff/Valid step (step: {inner_step})', val_metrics['rmse_diff'], step)

    def evaluate(self, batch_size: int, test_dataset: torch.utils.data.Dataset = None) -> tuple[float, dict]:
        self.valid_inner_steps = 0
        return super().evaluate(batch_size, test_dataset)

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        all_val_loss = torch.tensor(0.0).to(self.device)
        all_val_cosine = torch.tensor(0.0).to(self.device)
        all_val_stm_diff = torch.tensor(0.0).to(self.device)

        val_dataloader = self._valid_loader(batch_size)

        processed = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    processed += 1
                    val_loss = torch.tensor(0.0).to(self.device)
                    val_cosine = torch.tensor(0.0).to(self.device)
                    val_stm_diff = torch.tensor(0.0).to(self.device)

                    self.reset_stm()

                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']

                    number_of_inner_steps = len(interactions) + 1

                    for inner_step_idx in range(number_of_inner_steps):
                        self.valid_inner_steps += 1
                        if inner_step_idx == 0:
                            next_query, next_answer = self._move_multiple_batches(first_query, first_answer)
                        else:
                            next_query, next_answer = self._move_multiple_batches(
                                interactions[inner_step_idx - 1]['query'], interactions[inner_step_idx - 1]['answer'])

                        self._get_model().clone_reset_memory()
                        accumulated_stm = self._get_model().get_memory_state()

                        label_weights = self.label_weights(inner_step_idx) if callable(self.label_weights) else \
                            self.label_weights[inner_step_idx]

                        train_noises = self.noise_levels
                        self.noise_levels = (0.0, 0.0)

                        if self.use_amp:
                            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                                valid_batch = {
                                    **smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                                   pad_token_id=self.pad_token_id),
                                    'acc_stm': accumulated_stm,
                                }
                                loss, cosine_sim = self.compute_loss(valid_batch, weights=label_weights,
                                                                     inner_step_idx=inner_step_idx)
                        else:
                            valid_batch = {
                                **smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                               pad_token_id=self.pad_token_id),
                                'acc_stm': accumulated_stm,
                            }
                            loss, cosine_sim = self.compute_loss(valid_batch, weights=label_weights,
                                                                 inner_step_idx=inner_step_idx)

                        self.noise_levels = train_noises

                        val_loss += loss
                        val_cosine += cosine_sim

                        updated_stm = self._get_model().get_memory_state()

                        step_diff = torch.sqrt(F.mse_loss(updated_stm, accumulated_stm, reduction='mean'))
                        val_stm_diff += step_diff

                        if self.writer is not None:
                            self._valid_step_writer(self.valid_inner_steps, loss.item(), {
                                'cosine_sim': cosine_sim.item(),
                                'rmse_diff': step_diff.item(),
                                'loss': loss.item()
                            }, inner_step=inner_step_idx)

                    all_val_loss += val_loss / number_of_inner_steps
                    all_val_cosine += val_cosine / number_of_inner_steps
                    all_val_stm_diff += val_stm_diff / number_of_inner_steps

        avg_loss = all_val_loss / processed
        avg_cosine = all_val_cosine / processed
        avg_stm_diff = all_val_stm_diff / processed

        if self.use_ddp:
            avg_loss = distributed_mean(avg_loss)
            avg_cosine = distributed_mean(avg_cosine)
            avg_stm_diff = distributed_mean(avg_stm_diff)

        metrics = {
            'cosine_sim': avg_cosine.item(),
            'rmse_diff': avg_stm_diff.item(),
            'loss': avg_loss.item()
        }
        self.model.train()
        return avg_loss, metrics


class SupervisedMemoryAwareTrainer(BaseTrainer):
    """
    Supervised Memory Aware Trainer - made to train decoder memory cross-attention to use correct accumulated memory
    states, after basic self-supervised pre-training of memory attention network
    """

    def __init__(
            self,
            model: SupervisedMemoryAwareModel,
            device: torch.device,
            vocab_size: int,
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            max_seq_len: int = 256,
            pad_token_id: int = 0,
            train_only_decoder: bool = False,
            unfreeze_epochs: tuple[int, int] = (0, 0),
            use_system_prompt: bool = False,
            dataset_collate_fn: Callable[[list[Any]], dict[str, Any]] = MrlCurriculumDataset.collate_mrl_batch,
            **kwargs
    ):
        super(SupervisedMemoryAwareTrainer, self).__init__(
            model, device, dataset_collate_fn=dataset_collate_fn, **kwargs
        )
        self.vocab_size = vocab_size
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale
        self.get_batch_size = lambda batch: batch['query']['attention_mask'].size(0)
        self.total_inner_steps = 0
        self.valid_inner_steps = 0
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.train_only_decoder = train_only_decoder
        self.unfreeze_epochs = unfreeze_epochs
        self.use_system_prompt = use_system_prompt

        if not self.train_only_decoder:
            mem_attn_unfreeze_epoch, encoder_unfreeze_epoch = self.unfreeze_epochs
            if mem_attn_unfreeze_epoch != 0:
                self._get_model().memory_attention.freeze()

            if encoder_unfreeze_epoch != 0:
                self._get_model().encoder.freeze_all()

    def _get_model(self):
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = next(model.children())
        return model

    def reset_stm(self):
        self._get_model().reset_memory()

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

        if not self.train_only_decoder:
            mem_attn_unfreeze_epoch, encoder_unfreeze_epoch = self.unfreeze_epochs
            if mem_attn_unfreeze_epoch == epoch:
                self._get_model().memory_attention.unfreeze()

            if encoder_unfreeze_epoch == epoch:
                self._get_model().encoder.unfreeze_all(True, True) # TODO: Remove freeze memory after removing cross-attn from encoder

        self.accumulated_loss = torch.tensor(0.0, device=self.device)
        self.optimizer_step_count = 0
        accumulated_tokens = torch.tensor(0, dtype=torch.long, device=self.device)

        for batch_idx, batch in enumerate(dataloader):
            if not self.is_running:
                break
            else:
                if self.get_batch_size(batch) == batch_size:
                    self.total_steps += 1
                    self.epoch_steps = batch_idx

                    self.reset_stm()

                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']

                    number_of_inner_steps = len(interactions) + 1

                    prev_query, prev_answer = self._move_multiple_batches(first_query, first_answer)

                    for inner_step_idx in range(number_of_inner_steps):
                        self.total_inner_steps += 1

                        if inner_step_idx == 0:
                            next_query, next_answer = prev_query, prev_answer
                        else:
                            next_query, next_answer = self._move_multiple_batches(interactions[inner_step_idx - 1]['query'],
                                                                              interactions[inner_step_idx - 1]['answer'])

                        self._get_model().clone_reset_memory()

                        query_lens = next_query['attention_mask'].sum(dim=-1)

                        if self.use_amp:
                            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                                train_batch = {
                                    'prev': smart_concat(prev_query, prev_answer, max_length=self.max_seq_len,
                                                         pad_token_id=self.pad_token_id) if not self.use_system_prompt or inner_step_idx != 0 else self._move_batch(batch['system']),
                                    'next': smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                                         pad_token_id=self.pad_token_id),
                                }
                                accumulated_tokens += train_batch['next']['attention_mask'].sum()
                                loss, _ = self.compute_loss(train_batch, query_lens=query_lens, is_first_step=not self.use_system_prompt and inner_step_idx==0)
                        else:
                            train_batch = {
                                'prev': smart_concat(prev_query, prev_answer, max_length=self.max_seq_len,
                                                     pad_token_id=self.pad_token_id) if not self.use_system_prompt or inner_step_idx != 0 else self._move_batch(batch['system']),
                                'next': smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                                     pad_token_id=self.pad_token_id),
                            }
                            accumulated_tokens += train_batch['next']['attention_mask'].sum()
                            loss, _ = self.compute_loss(train_batch, query_lens=query_lens, is_first_step=not self.use_system_prompt and inner_step_idx==0)

                        self.accumulated_loss += loss
                        loss = loss / self.gradient_accumulation_steps

                        if self.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        self.optimizer_step_count += 1
                        if self.optimizer_step_count % self.gradient_accumulation_steps == 0:
                            # Clip gradients after accumulation
                            if self.use_amp:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self._get_model().trainable_parameters(), max_norm=1.0,
                                                           error_if_nonfinite=False)
                            if self.use_amp:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()

                            optimizer.zero_grad()

                            if scheduler is not None:
                                scheduler.step()

                            if self.writer and self.total_steps % self.tensorboard_interval:
                                loss_item = self.accumulated_loss / self.gradient_accumulation_steps
                                self._train_writer(
                                    loss_item.item(),
                                    epoch_step=(batch_idx * number_of_inner_steps) + inner_step_idx,
                                    inner_step=inner_step_idx,
                                )

                            self.accumulated_loss = torch.tensor(0.0, device=self.device)
                            self.optimizer_step_count = 0

                        prev_query, prev_answer = self._move_multiple_batches(next_query, next_answer)

                        if self.writer and self.total_steps % self.tensorboard_interval:
                            self.total_tokens += accumulated_tokens.item()
                            accumulated_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
                            self.writer.add_scalar(
                                'Processed tokens',
                                self.total_tokens,
                                self.total_inner_steps
                            )

                        for callback in self.callbacks:
                            should_stop = callback.on_batch_end(self.model,
                                                                (batch_idx * number_of_inner_steps) + inner_step_idx,
                                                                loss, train_batch['next'])
                            if should_stop:
                                self.is_running = False

        if self.validation_dataset:
            self.validation_steps = 0
            self.valid_inner_steps = 0
            val_loss, val_metrics = self.validate(batch_size)
            if self.use_ddp:
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

    def _move_batch(self, batch: TokenizedDict) -> TokenizedDict:
        if self.use_amp:
            return {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
            }
        else:
            return {
                'input_ids': batch['input_ids'].to(self.device, dtype=self.dtype),
                'attention_mask': batch['attention_mask'].to(self.device, dtype=self.dtype),
            }

    def _move_multiple_batches(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._move_batch(batch) for batch in batches]

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        model = self._get_model()

        router_loss = model.decoder.model.moe_router_loss()
        loss = main_loss + self.moe_aux_loss_scale * router_loss

        if self.writer is not None:
            if self.model.training:
                self.writer.add_scalar('Router aux loss/Train', router_loss.item(), self.total_steps)
                self.writer.add_scalar('Model loss/Train', main_loss.item(), self.total_steps)
            else:
                self.writer.add_scalar('Router aux loss/Valid', router_loss.item(), self.total_steps)
                self.writer.add_scalar('Model loss/Valid', main_loss.item(), self.total_steps)

        return loss

    def compute_loss(self, batch: dict[str, dict[str, torch.Tensor]], query_lens: torch.Tensor = None, is_first_step: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_inputs = batch['prev']['input_ids']
        encoder_mask = batch['prev']['attention_mask']
        decoder_inputs = batch['next']['input_ids']
        decoder_targets = batch['next']['input_ids']
        decoder_mask = batch['next']['attention_mask']

        decoder_logits = self.model(
            encoder_inputs,
            decoder_inputs,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            is_first_step=is_first_step,
        )

        shifted_logits = decoder_logits[:, :-1].contiguous()
        shifted_targets = decoder_targets[:, 1:].contiguous()

        for i in range(shifted_targets.size(0)):
            end = query_lens[i].item()
            shifted_targets[i, :end] = -100

        decoder_loss = F.cross_entropy(
            shifted_logits.view(-1, self.vocab_size),
            shifted_targets.view(-1),
            ignore_index=-100,
        )

        decoder_loss = self._moe_aux_loss(decoder_loss)

        return decoder_loss, decoder_logits

    def _train_writer(self, loss: float, epoch_step: int, inner_step: int) -> None:
        self.writer.add_scalar(
            'Loss/train',
            loss,
            self.total_inner_steps,
        )
        self.writer.add_scalar(
            'Loss/train last epoch',
            loss,
            epoch_step,
        )
        self.writer.add_scalar(
            'Perplexity/train',
            torch.exp(torch.tensor(loss)).item(),
            self.total_inner_steps,
        )

        self.writer.add_scalar(
            f'Loss/train (step: {inner_step})',
            loss,
            self.total_inner_steps,
        )
        self.writer.add_scalar(
            f'Perplexity/train (step: {inner_step})',
            torch.exp(torch.tensor(loss)).item(),
            self.total_inner_steps,
        )

    def _valid_writer(self, epoch: int, val_loss: float, val_metrics: dict):
        self.writer.add_scalar('Loss/Valid', val_loss, epoch)
        self.writer.add_scalar('Perplexity/Valid', math.exp(val_loss), epoch)
        if val_metrics['accuracy']:
            self.writer.add_scalar('Decoder node accuracy/Valid', val_metrics['node_accuracy'], epoch)
            self.writer.add_scalar('Decoder avg. accuracy/Valid', val_metrics['accuracy'], epoch)

    def _valid_step_writer(self, step: int, val_loss: float, val_metrics: dict, inner_step: int):
        self.writer.add_scalar('Loss/Valid step', val_loss, step)
        self.writer.add_scalar('Perplexity/Valid step', math.exp(val_loss), step)

        self.writer.add_scalar(f'Loss/Valid step (step: {inner_step})', val_loss, step)
        self.writer.add_scalar(f'Perplexity/Valid step (step: {inner_step})', math.exp(val_loss), step)

        if val_metrics['accuracy']:
            self.writer.add_scalar('Decoder node accuracy/Valid step', val_metrics['node_accuracy'], step)
            self.writer.add_scalar('Decoder avg. accuracy/Valid', val_metrics['accuracy'], step)

            self.writer.add_scalar(f'Decoder node accuracy/Valid step (step: {inner_step})',
                                   val_metrics['node_accuracy'], step)
            self.writer.add_scalar(f'Decoder avg. accuracy/Valid (step: {inner_step})', val_metrics['accuracy'], step)

    def evaluate(self, batch_size: int, test_dataset: torch.utils.data.Dataset = None) -> tuple[float, dict]:
        self.valid_inner_steps = 0
        return super().evaluate(batch_size, test_dataset)

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        all_val_loss = torch.tensor(0.0).to(self.device)
        correct_alm = torch.tensor(0).to(self.device)
        total_alm = torch.tensor(0).to(self.device)

        val_dataloader = self._valid_loader(batch_size)

        processed = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    processed += 1
                    val_loss = torch.tensor(0.0).to(self.device)

                    self.reset_stm()

                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']

                    number_of_inner_steps = len(interactions) + 1

                    prev_query, prev_answer = self._move_multiple_batches(first_query, first_answer)

                    for inner_step_idx in range(number_of_inner_steps):
                        self.total_inner_steps += 1

                        if inner_step_idx == 0:
                            next_query, next_answer = prev_query, prev_answer
                        else:
                            next_query, next_answer = self._move_multiple_batches(interactions[inner_step_idx - 1]['query'],
                                                                              interactions[inner_step_idx - 1]['answer'])

                        self._get_model().clone_reset_memory()

                        query_lens = next_query['attention_mask'].sum(dim=-1)

                        if self.use_amp:
                            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                                valid_batch = {
                                    'prev': smart_concat(prev_query, prev_answer, max_length=self.max_seq_len,
                                                         pad_token_id=self.pad_token_id) if not self.use_system_prompt or inner_step_idx != 0 else self._move_batch(batch['system']),
                                    'next': smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                                         pad_token_id=self.pad_token_id),
                                }
                                decoder_loss, decoder_logits = self.compute_loss(valid_batch, query_lens=query_lens, is_first_step=not self.use_system_prompt and inner_step_idx==0)
                        else:
                            valid_batch = {
                                'prev': smart_concat(prev_query, prev_answer, max_length=self.max_seq_len,
                                                     pad_token_id=self.pad_token_id) if not self.use_system_prompt or inner_step_idx != 0 else self._move_batch(batch['system']),
                                'next': smart_concat(next_query, next_answer, max_length=self.max_seq_len,
                                                     pad_token_id=self.pad_token_id),
                            }
                            decoder_loss, decoder_logits = self.compute_loss(valid_batch, query_lens=query_lens, is_first_step=not self.use_system_prompt and inner_step_idx==0)

                        val_loss += decoder_loss

                        prev_query, prev_answer = self._move_multiple_batches(next_query, next_answer)

                        shifted_logits = decoder_logits[:, :-1].contiguous()
                        shifted_targets = valid_batch['next']['input_ids'][:, 1:].to(self.device).contiguous()

                        for i in range(shifted_targets.size(0)):
                            end = query_lens[i].item()
                            shifted_targets[i, :end] = -100

                        valid_alm_indices = shifted_targets != -100

                        if valid_alm_indices.any():
                            preds_alm = shifted_logits.argmax(-1)
                            correct_in_step = (preds_alm[valid_alm_indices] == shifted_targets[valid_alm_indices]).sum()
                            total_in_step = valid_alm_indices.sum()
                            correct_alm += correct_in_step
                            total_alm += total_in_step

                            step_acc = (correct_in_step / total_in_step * 100).item() if total_in_step > 0 else 0.0
                        else:
                            step_acc = 0.0

                        if self.use_ddp:
                            avg_step_acc = distributed_value_mean(step_acc, device=self.device)
                        else:
                            avg_step_acc = step_acc

                        if self.writer is not None:
                            self._valid_step_writer(self.valid_inner_steps, decoder_loss.item(), {
                                'accuracy': avg_step_acc,
                                'node_accuracy': step_acc,
                                'loss': decoder_loss.item()
                            }, inner_step=inner_step_idx)

                    all_val_loss += val_loss / number_of_inner_steps

        avg_loss = all_val_loss / processed
        alm_acc = (correct_alm / total_alm * 100) if total_alm > 0 else torch.tensor(0.0).to(self.device)
        node_alm_acc = alm_acc.item()

        if self.use_ddp:
            avg_loss = distributed_mean(avg_loss)
            alm_acc = distributed_mean(alm_acc)

        metrics = {
            'accuracy': alm_acc.item(),
            'node_accuracy': node_alm_acc,
            'loss': avg_loss.item()
        }
        self.model.train()
        return avg_loss, metrics
