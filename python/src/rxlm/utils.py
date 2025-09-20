import random, gc
import torch
import numpy as np

def human_format(num: int):
    """Format numbers to human-readable format."""
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def get_model_size(model: torch.nn.Module):
    """Calculate all models parameters with requires_grad param set as True"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'Model params {human_format(trainable_params)}'

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.

    Applied on 3 libs: PyTorch, Numpy and random

    seed (int): Random seed value
    """
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def cache_clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
