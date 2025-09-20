import torch
import torch.distributed as dist
import os
from ..utils import set_random_seed

def get_os_ddp_config():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    return rank, world_size

def distributed_mean(x: torch.Tensor) -> torch.Tensor:
    """Average tensor across all devices"""
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x

def distributed_value_mean(value: float, device: torch.device = None) -> float:
    """Average float value across all devices"""
    tensor = torch.tensor(value, device=device)
    reduced = distributed_mean(tensor)
    return reduced.item()

def set_distributed_random_seed(seed: int):
    rank = dist.get_rank() if dist.is_initialized() else get_os_ddp_config()[0]
    set_random_seed(seed + rank)
