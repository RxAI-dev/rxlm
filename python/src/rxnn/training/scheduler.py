import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

def get_transformer_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0
):
    if warmup_steps > 0:
        # Warmup + cosine decay
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            remaining = max(0, current_step - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * remaining / (num_training_steps - warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)
    else:
        return CosineAnnealingLR(optimizer, T_max=num_training_steps)

def calculate_steps(
        dataset_size: int,
        epochs: int,
        batch_size: int,
        warmup_ratio: float = 0.0,
        num_workers: int = 1,
        gradient_accumulation_steps: int = 1,
        verbose: bool = True,
):
    steps_per_epoch = int((dataset_size / batch_size - 1) // num_workers)
    total_steps = int((epochs * steps_per_epoch) / gradient_accumulation_steps)
    warmup_steps = int(warmup_ratio * total_steps)
    if verbose:
        print(f'Total steps: {total_steps}')
        print(f'Warmup steps: {warmup_steps}')
        print(f'Total steps per epoch: {steps_per_epoch}')
    return { 'total': total_steps, 'warmup': warmup_steps, 'epoch': steps_per_epoch}

def calculate_steps_for_smst(
        dataset_size: int,
        epochs: int,
        curriculum_steps: int,
        batch_size: int,
        warmup_ratio: float = 0.0,
        num_workers: int = 1,
        gradient_accumulation_steps: int = 1,
        verbose: bool = True,
):
    batches_per_epoch = int(((dataset_size / batch_size - 1) // num_workers))
    inner_steps_per_epoch = int(batches_per_epoch * curriculum_steps)

    total_steps = int((epochs * inner_steps_per_epoch) / gradient_accumulation_steps)
    warmup_steps = int(warmup_ratio * total_steps)
    if verbose:
        print(f'Total steps: {total_steps}')
        print(f'Warmup steps: {warmup_steps}')
        print(f'Total batches per epoch: {batches_per_epoch}')
        print(f'Total steps per epoch: {inner_steps_per_epoch}')
    return { 'total': total_steps, 'warmup': warmup_steps, 'epoch': inner_steps_per_epoch, 'batch': batches_per_epoch }

