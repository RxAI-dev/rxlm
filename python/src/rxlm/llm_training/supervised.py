import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import math
from typing import Union
from sklearn.metrics import f1_score
from ..training.base import BaseTrainer
from .models import MLMTrainingModel
from ..training.ddp import distributed_mean

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
            model: torch.nn.Module,
            device: torch.device,
            vocab_size: int,
            use_amp: bool = False,
            dtype: torch.dtype = None,
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            use_f1_metrics: bool = False,
            is_sft: bool = False,
            **kwargs
    ):
        super(AutoregressiveTrainer, self).__init__(model, device, use_amp=use_amp, dtype=dtype,
                                                    target_field_name='targets', **kwargs)
        self.vocab_size = vocab_size
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale
        self.use_f1_metrics = use_f1_metrics
        self.is_sft = is_sft

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

        if self.is_sft:
            loss = F.cross_entropy(
                shifted_logits.view(-1, self.vocab_size),
                shifted_targets.view(-1),
                ignore_index=-100
            )
        else:
            loss = F.cross_entropy(
                shifted_logits.view(-1, self.vocab_size),
                shifted_targets.view(-1),
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

        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

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

                        if self.use_f1_metrics:
                            # Collect predictions and labels for F1 score
                            all_preds.append(preds[valid_indices].cpu())
                            all_labels.append(shifted_targets[valid_indices].cpu())

        avg_loss = (val_loss / len(val_dataloader)).item()
        acc = (correct / total * 100) if total > 0 else torch.tensor(0.0).to(self.device)
        node_acc = acc.item()
        if self.use_ddp:
            acc = distributed_mean(acc)

        f1_macro = 0.0
        f1_weighted = 0.0

        if self.use_f1_metrics:
            if self.use_ddp:
                local_preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
                local_labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)

                gathered_preds = [torch.zeros_like(local_preds) for _ in range(torch.distributed.get_world_size())]
                gathered_labels = [torch.zeros_like(local_labels) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_preds, local_preds)
                torch.distributed.all_gather(gathered_labels, local_labels)

                all_preds_tensor = torch.cat(gathered_preds).cpu().numpy()
                all_labels_tensor = torch.cat(gathered_labels).cpu().numpy()
            else:
                all_preds_tensor = torch.cat(all_preds).cpu().numpy() if all_preds else np.array([])
                all_labels_tensor = torch.cat(all_labels).cpu().numpy() if all_labels else np.array([])


            if len(all_labels_tensor) > 0:
                f1_macro = f1_score(all_labels_tensor, all_preds_tensor, average='macro')
                f1_weighted = f1_score(all_labels_tensor, all_preds_tensor, average='weighted')

                if self.writer is not None and (not self.use_ddp or torch.distributed.get_rank() == 0):
                    self.writer.add_scalar('F1 Macro/Valid total', f1_macro, self.total_validation_steps)
                    self.writer.add_scalar('F1 Weighted/Valid total', f1_weighted, self.total_validation_steps)

        if self.use_f1_metrics:
            metrics = {
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'accuracy': acc.item(),
                'node_accuracy': node_acc,
            }
        else:
            metrics = {
                'accuracy': acc.item(),
                'node_accuracy': node_acc,
            }
        self.model.train()
        return avg_loss, metrics
