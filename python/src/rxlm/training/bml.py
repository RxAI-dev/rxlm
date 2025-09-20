import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import math
from typing import Union
from ..transformers.models import ReactiveTransformerDecoder
from ..training.base import BaseTrainer
from .models import MLMTrainingModel, JointTrainingModel
from .ddp import distributed_mean


class MLMTrainer(BaseTrainer):
    def __init__(
            self,
            model: MLMTrainingModel,
            device: torch.device,
            vocab_size: int,
            use_amp: bool = False,
            dtype: torch.dtype = None,
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            **kwargs
    ):
        super(MLMTrainer, self).__init__(model, device, use_amp=use_amp, dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self.model(
            inputs,
            attention_mask=attention_mask
        )

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        return self._moe_aux_loss(loss), logits

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        model = next(self.model.children()) if isinstance(self.model, DistributedDataParallel) else self.model

        router_loss = model.encoder.model.moe_router_loss()
        loss = main_loss + self.moe_aux_loss_scale * router_loss

        if self.writer is not None:
            if self.model.training:
                self.writer.add_scalar('Router aux loss/Train', router_loss.item(), self.total_steps)
                self.writer.add_scalar('Model loss/Train', main_loss.item(), self.total_steps)
            else:
                self.writer.add_scalar('Router aux loss/Valid', router_loss.item(), self.total_validation_steps)
                self.writer.add_scalar('Model loss/Valid', main_loss.item(), self.total_validation_steps)

        return loss

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        val_dataloader = self._valid_loader(batch_size)
        val_loss = torch.tensor(0.0).to(self.device)
        correct = torch.tensor(0).to(self.device)
        total = torch.tensor(0).to(self.device)

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    self.total_validation_steps += 1
                    self.validation_steps += 1
                    loss, logits = self.valid_step(batch)
                    val_loss += loss
                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Valid total', loss.item(), self.total_validation_steps)
                        self.writer.add_scalar('Perplexity/Valid', torch.exp(loss).item(), self.total_validation_steps)

                    labels = batch[self.target_field_name].to(self.device)
                    valid_indices = labels != -100
                    if valid_indices.any():
                        preds = logits.argmax(-1)
                        batch_correct = (preds[valid_indices] == labels[valid_indices]).sum()
                        batch_total = valid_indices.sum()
                        batch_acc = (batch_correct / batch_total * 100) if total > 0 else torch.tensor(0.0).to(
                            self.device)
                        if self.writer is not None:
                            self.writer.add_scalar('Accuracy/Valid total', batch_acc.item(),
                                                   self.total_validation_steps)
                        correct += batch_correct
                        total += batch_total

        avg_loss = (val_loss / len(val_dataloader)).item()
        acc = (correct / total * 100) if total > 0 else torch.tensor(0.0).to(self.device)
        node_acc = acc.item()
        if self.use_ddp:
            acc = distributed_mean(acc)

        metrics = {
            'accuracy': acc.item(),
            'node_accuracy': node_acc,
        }
        self.model.train()
        return avg_loss, metrics


class AutoregressiveTrainer(BaseTrainer):
    def __init__(
            self,
            model: ReactiveTransformerDecoder,
            device: torch.device,
            vocab_size: int,
            use_amp: bool = False,
            dtype: torch.dtype = None,
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            **kwargs
    ):
        super(AutoregressiveTrainer, self).__init__(model, device, use_amp=use_amp, dtype=dtype,
                                                    target_field_name='targets', **kwargs)
        self.vocab_size = vocab_size
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']

        outputs = self.model(
            inputs,
            attention_mask=attention_mask
        )

        shifted_logits = outputs[:, :-1].contiguous()
        shifted_targets = targets[:, 1:].contiguous()

        loss = F.cross_entropy(
            shifted_logits.view(-1, self.vocab_size),
            shifted_targets.view(-1)
        )

        return self._moe_aux_loss(loss), outputs

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        model = next(self.model.children()) if isinstance(self.model, DistributedDataParallel) else self.model

        router_loss = model.model.moe_router_loss()
        loss = main_loss + self.moe_aux_loss_scale * router_loss

        if self.writer is not None:
            if self.model.training:
                self.writer.add_scalar('Router aux loss/Train', router_loss.item(), self.total_steps)
                self.writer.add_scalar('Model loss/Train', main_loss.item(), self.total_steps)
            else:
                self.writer.add_scalar('Router aux loss/Valid', router_loss.item(), self.total_steps)
                self.writer.add_scalar('Model loss/Valid', main_loss.item(), self.total_steps)

        return loss

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        val_dataloader = self._valid_loader(batch_size)
        val_loss = torch.tensor(0.0).to(self.device)
        correct = torch.tensor(0).to(self.device)
        total = torch.tensor(0).to(self.device)

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    self.total_validation_steps += 1
                    self.validation_steps += 1
                    loss, logits = self.valid_step(batch)
                    val_loss += loss
                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Valid total', loss.item(), self.total_validation_steps)
                        self.writer.add_scalar('Perplexity/Valid', torch.exp(loss).item(), self.total_validation_steps)
                    shifted_logits = logits[:, :-1].contiguous()
                    shifted_targets = batch[self.target_field_name][:, 1:].to(self.device).contiguous()
                    valid_indices = shifted_targets != -100
                    if valid_indices.any():
                        preds = shifted_logits.argmax(-1)
                        batch_correct = (preds[valid_indices] == shifted_targets[valid_indices]).sum()
                        batch_total = valid_indices.sum()
                        batch_acc = (batch_correct / batch_total * 100) if total > 0 else torch.tensor(0.0).to(
                            self.device)
                        if self.writer is not None:
                            self.writer.add_scalar('Accuracy/Valid total', batch_acc.item(),
                                                   self.total_validation_steps)
                        correct += batch_correct
                        total += batch_total

        avg_loss = (val_loss / len(val_dataloader)).item()
        acc = (correct / total * 100) if total > 0 else torch.tensor(0.0).to(self.device)
        node_acc = acc.item()
        if self.use_ddp:
            acc = distributed_mean(acc)

        metrics = {
            'accuracy': acc.item(),
            'node_accuracy': node_acc,
        }
        self.model.train()
        return avg_loss, metrics


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
            **kwargs
    ):
        super(JointLMTrainer, self).__init__(model, device, use_amp=use_amp, dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.components_loss_log_interval = components_loss_log_interval
        self.encoder_loss_scale = encoder_loss_scale
        self.decoder_loss_scale = decoder_loss_scale
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale

    def train_step(self, batch: dict[str, Union[torch.Tensor, dict[torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        if self.use_amp:
            batch = {
                k: (
                    { kk: vv.to(self.device) for kk, vv in v.items() } if not torch.is_tensor(v) else v.to(self.device)
                ) for k, v in batch.items()
            }
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                (encoder_loss, decoder_loss), _ = self.compute_loss(batch)
        else:
            batch = {
                k: (
                    {
                        kk: vv.to(self.device, dtype=self.dtype) for kk, vv in v.items()
                    } if not torch.is_tensor(v) else v.to(self.device, dtype=self.dtype)
                ) for k, v in batch.items()
            }
            (encoder_loss, decoder_loss), _ = self.compute_loss(batch)

        if self.components_loss_log_interval is not None:
            if batch_idx % self.components_loss_log_interval == 0:
                print(f"Encoder loss: {encoder_loss.item():.4f}")
                print(f"Decoder loss: {decoder_loss.item():.4f}")
                if self.encoder_loss_scale != 1.0:
                    print(
                        f"Encoder loss scaled by {self.encoder_loss_scale}: {(encoder_loss * self.encoder_loss_scale).item() :.4f}")
                if self.decoder_loss_scale != 1.0:
                    print(
                        f"Decoder loss scaled by {self.decoder_loss_scale}: {(decoder_loss * self.decoder_loss_scale).item() :.4f}")

        return (encoder_loss * self.encoder_loss_scale) + (decoder_loss * self.decoder_loss_scale)

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        model = next(self.model.children()) if isinstance(self.model, DistributedDataParallel) else self.model

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
            attention_mask=attention_mask
        )

        encoder_loss = F.cross_entropy(
            encoder_logits.view(-1, self.vocab_size),
            encoder_labels.view(-1),
            ignore_index=-100
        )

        shifted_logits = decoder_logits[:, :-1].contiguous()
        shifted_targets = decoder_targets[:, 1:].contiguous()

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
                    if self.use_amp:
                        batch = {
                            k: ({kk: vv.to(self.device) for kk, vv in v.items()} if not torch.is_tensor(v) else v.to(
                                self.device)) for k, v in batch.items()}
                        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                            (encoder_loss, decoder_loss), (encoder_logits, decoder_logits) = self.compute_loss(batch)
                    else:
                        batch = {k: (
                            {kk: vv.to(self.device, dtype=self.dtype) for kk, vv in v.items()} if not torch.is_tensor(
                                v) else v.to(self.device, dtype=self.dtype)) for k, v in batch.items()}
                        (encoder_loss, decoder_loss), (encoder_logits, decoder_logits) = self.compute_loss(batch)

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
