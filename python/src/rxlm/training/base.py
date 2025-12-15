import torch
import math
import os

from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Callable, Any
from .callbacks import TrainerCallback, ModelSaveCallback, JointModelSaveCallback
from .ddp import get_os_ddp_config, distributed_value_mean


class BaseTrainer(ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            optimizer: torch.optim.Optimizer = None,
            dataset: torch.utils.data.Dataset = None,
            validation_dataset: torch.utils.data.Dataset = None,
            callbacks: list[TrainerCallback] = None,
            log_dir: str = None,
            use_ddp: bool = False,
            use_amp: bool = False,
            dtype: torch.dtype = None,
            target_field_name: str = 'labels',
            get_batch_size: Callable[[dict], int] = None,
            gradient_accumulation_steps: int = 1,
            tensorboard_interval: int = 10,
            dataset_collate_fn: Callable[[list[Any]], dict[str, Any]] = None,
            use_iterable_dataset: bool = False,
            num_dataloader_workers: int = 0,
            ddp_shuffle: bool = True,
            use_te_fp8: bool = False,
            fp8_history_len: int = 256,
            fp8_margin: int = 0,
    ):
        if get_batch_size is None:
            self.get_batch_size = lambda batch: batch['attention_mask'].size(0)
        else:
            self.get_batch_size = get_batch_size
        if use_amp:
            self.model = model.to(device)
        else:
            self.model = model.to(device, dtype=dtype)
        self.device = device
        self.optimizer = optimizer
        self.dataset = dataset
        self.callbacks = callbacks or []
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.use_ddp = use_ddp
        self.use_amp = use_amp
        self.dtype = dtype
        self.is_running = False
        self.validation_dataset = validation_dataset
        self.best_val_loss = float('inf')
        self.validation_metrics = {}
        self.target_field_name = target_field_name
        self.total_tokens = 0
        self.total_steps = 0
        self.validation_steps = 0
        self.total_validation_steps = 0
        self.epoch_steps = 0
        self.current_epoch = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accumulated_loss = 0.0
        self.optimizer_step_count = 0
        self.tensorboard_interval = tensorboard_interval
        self.dataset_collate_fn = dataset_collate_fn
        self.use_iterable_dataset = use_iterable_dataset
        self.num_dataloader_workers = num_dataloader_workers
        self.ddp_shuffle = ddp_shuffle
        self.use_te_fp8 = use_te_fp8
        self.fp8_history_len = fp8_history_len
        self.fp8_margin = fp8_margin
        if use_te_fp8:
            from transformer_engine.common import recipe
            self.use_amp = False

            self.fp8_recipe = recipe.DelayedScaling(
                fp8_format=recipe.Format.HYBRID,
                amax_history_len=fp8_history_len,
                amax_compute_algo='max',
                margin=fp8_margin,
            )
        else:
            self.fp8_recipe = None

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def train_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                loss, _ = self.compute_loss(batch)
        elif self.use_te_fp8 and self.fp8_recipe:
            import transformer_engine.pytorch as te
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                loss, _ = self.compute_loss(batch)
        else:
            loss, _ = self.compute_loss(batch)
        return loss

    def __call__(
            self,
            epochs: int,
            batch_size: int,
            dataset: torch.utils.data.Dataset = None,
            validation_dataset: torch.utils.data.Dataset = None,
            optimizer: torch.optim.Optimizer = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            ddp_find_unused_parameters: bool = False,
    ) -> None:
        self.is_running = True
        if dataset is None:
            assert self.dataset is not None, 'You have to specify a dataset for training'
            dataset = self.dataset
        if optimizer is None:
            assert self.optimizer is not None, 'You have to specify an optimizer for training'
            optimizer = self.optimizer
        # Set validation dataset
        if validation_dataset is not None:
            self.validation_dataset = validation_dataset

        if self.use_ddp:
            rank, world_size = get_os_ddp_config()
            self.model = DistributedDataParallel(self.model, device_ids=[self.device.index], find_unused_parameters=ddp_find_unused_parameters)
            train_sampler = torch.utils.data.DistributedSampler(dataset, shuffle=not self.use_iterable_dataset and self.ddp_shuffle, rank=rank, num_replicas=world_size, drop_last=True) if not self.use_iterable_dataset else None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                pin_memory=True,
                collate_fn=self.dataset_collate_fn,
                num_workers=self.num_dataloader_workers,
            )
        else:
            train_sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=not self.use_iterable_dataset,
                pin_memory=True,
                collate_fn=self.dataset_collate_fn,
                drop_last=True,
                num_workers=self.num_dataloader_workers,
            )

        scaler = torch.amp.GradScaler() if self.use_amp and self.dtype != torch.bfloat16 else None

        self.model.train()
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            if not self.is_running:
                break
            else:
                self.current_epoch = epoch
                self.epoch_steps = 0
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                self._run_epoch(dataloader, epoch, optimizer, batch_size, scaler=scaler, scheduler=scheduler)
                if self.use_ddp:
                    dist.barrier()

        if self.use_ddp:
            dist.destroy_process_group()
        self.is_running = False
        self.model.eval()
        self.on_training_end()

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

        for batch_idx, batch in enumerate(dataloader):
            if not self.is_running:
                break
            else:
                self.total_steps += 1
                self.epoch_steps = batch_idx
                accumulated_tokens += batch['attention_mask'].sum()
                loss = self.train_step(batch, batch_idx)
                self.accumulated_loss += loss
                loss = loss / self.gradient_accumulation_steps

                if self.use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

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


                    if self.writer and self.total_steps % self.tensorboard_interval == 0:
                        loss_item = (self.accumulated_loss / self.gradient_accumulation_steps).item()
                        self.writer.add_scalar(
                            'Loss/train',
                            loss_item,
                            self.total_steps,
                        )
                        self.writer.add_scalar(
                            'Loss/train last epoch',
                            loss_item,
                            batch_idx
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

                    self.accumulated_loss = torch.tensor(0.0, device=self.device)
                    self.optimizer_step_count = 0

                for callback in self.callbacks:
                    should_stop = callback.on_batch_end(self.model, batch_idx, loss, batch)
                    if should_stop:
                        self.is_running = False

        if self.validation_dataset:
            self.validation_steps = 0
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

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end(self.model)
        if self.writer:
            self.writer.close()

    def _valid_writer(self, epoch: int, val_loss: float, val_metrics: dict):
        self.writer.add_scalar('Loss/Valid', val_loss, epoch)
        self.writer.add_scalar('Perplexity/Valid', math.exp(val_loss), epoch)
        if val_metrics['accuracy']:
            self.writer.add_scalar('Node Accuracy/Valid', val_metrics['node_accuracy'], epoch)
            self.writer.add_scalar('Avg. Accuracy/Valid', val_metrics['accuracy'], epoch)

    def valid_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                loss, outputs = self.compute_loss(batch)
        elif self.use_te_fp8 and self.fp8_recipe:
            import transformer_engine.pytorch as te
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                loss, outputs = self.compute_loss(batch)
        else:
            loss, outputs = self.compute_loss(batch)
        return loss, outputs

    def _valid_loader(self, batch_size: int):
        val_dataset = self.validation_dataset
        if self.use_ddp:
            val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
            return torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=self.dataset_collate_fn,
            )
        else:
            return torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.dataset_collate_fn,
            )

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        val_dataloader = self._valid_loader(batch_size)
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    self.validation_steps += 1
                    self.total_validation_steps += 1
                    loss, outputs = self.valid_step(batch)
                    val_loss += loss.item()

        avg_loss = val_loss / len(val_dataloader)
        metrics = {}
        self.model.train()
        return avg_loss, metrics

    def evaluate(self, batch_size: int, test_dataset: torch.utils.data.Dataset = None) -> tuple[float, dict]:
        valid_dataset = self.validation_dataset
        if test_dataset is not None:
            self.validation_dataset = test_dataset
        else:
            assert self.validation_dataset is not None, 'Test or validation dataset have to be provided for evaluation'

        self.validation_steps = 0
        val_loss, val_metrics = self.validate(batch_size)
        if self.use_ddp:
            val_loss = distributed_value_mean(val_loss, device=self.device)

        # Filter model save callbacks to not duplicate models
        for callback in [
            cb for cb in self.callbacks if not (
                isinstance(cb, JointModelSaveCallback) or isinstance(cb, ModelSaveCallback)
            )
        ]:
            callback.on_validation_end(self.model, -1, val_loss, val_metrics)

        self.validation_dataset = valid_dataset

        return val_loss, val_metrics