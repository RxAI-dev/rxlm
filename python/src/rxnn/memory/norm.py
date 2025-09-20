import torch
import torch.nn as nn
from typing import TypedDict

class AdaptivePositionalMemoryNorm(nn.Module):
    def __init__(
        self,
        num_slots: int,
        dim: int,
        decay: float = 0.9,
        use_scale: bool = True,
        use_gate: bool = True,
        init_gate: float = -2.0,
        per_dim_scale: bool = False,
    ):
        super(AdaptivePositionalMemoryNorm, self).__init__()
        self.use_gate = use_gate
        self.num_slots = num_slots
        self.dim = dim
        self.decay = decay
        self.eps = 1e-6

        # Learnable parameters
        scale_shape = (num_slots, 1) if not per_dim_scale else (dim,)
        self.scale = nn.Parameter(torch.ones(*scale_shape)) if use_scale else None
        self.gate = nn.Parameter(torch.full((num_slots, 1), init_gate)) if use_gate else None

        # EMA buffers
        self.register_buffer("ema_rms", torch.ones(num_slots, 1))

        # Initialize parameters
        if self.scale is not None:
            nn.init.normal_(self.scale, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate current RMS per slot
        # x: [batch_size, num_slots, dim]
        current_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()  # [batch, num_slots, 1]
        slot_rms = current_rms.mean(dim=0)  # [num_slots, 1] (average over batch)

        # Update EMA during training
        if self.training:
            self.ema_rms = self.decay * self.ema_rms + (1 - self.decay) * slot_rms.detach() # [num_slots, 1]

        # Normalize using EMA statistics
        x_norm = x * torch.rsqrt(self.ema_rms + self.eps) # [batch_size, num_slots, dim] * [num_slots, 1]

        # Apply learned scale per slot
        if self.scale is not None:
            x_norm = x_norm * self.scale # [batch_size, num_slots, dim] * [num_slots, 1] or [dim]

        # Apply gating mechanism
        if self.use_gate:
            gate = torch.sigmoid(self.gate)  # [num_slots, 1]
            return gate * x_norm + (1 - gate) * x # [batch_size, num_slots, dim] * [num_slots, 1]

        return x_norm

class AdaptiveRMSMemoryNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        use_gate: bool = True,
        decay: float = 0.99,
        init_scale: float = 1.0,
        init_gate: float = -4.0  # Start with gate closed (no normalization)
    ):
        super().__init__()
        self.use_gate = use_gate
        self.scale = nn.Parameter(torch.ones(dim) * init_scale)
        self.gate = nn.Parameter(torch.tensor([init_gate]))  # Scalar gate for this layer
        self.eps = 1e-6
        self.decay = decay
        self.register_buffer("ema_rms", torch.ones(1))  # Scalar EMA RMS for the entire layer's STM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_slots, dim]
        if self.training and hasattr(self, 'ema_rms'):
            # Compute current RMS across all slots and batch (scalar)
            current_rms = x.pow(2).mean(dim=-1).mean().sqrt()
            self.ema_rms = self.ema_rms * self.decay + current_rms * (1 - self.decay)
            rms = self.ema_rms
        else:
            # Compute RMS per slot (mean over dim)
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()  # [batch_size, num_slots, 1]

        # Normalize each slot's embedding vector
        normalized = x * torch.rsqrt(rms + self.eps)
        normalized = normalized * self.scale  # Apply per-dimension scaling

        if self.use_gate:
            gate_factor = torch.sigmoid(self.gate)  # Scalar gate (0-1)
            return normalized * gate_factor + x * (1 - gate_factor)
        else:
            return normalized

class SimpleRMSMemoryNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        use_gate: bool = True,
        init_scale: float = 1.0,
        init_gate: float = -4.0
    ):
        super().__init__()
        self.use_gate = use_gate
        self.scale = nn.Parameter(torch.ones(dim) * init_scale)
        self.gate = nn.Parameter(torch.tensor([init_gate]))  # Scalar gate
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()  # [batch_size, num_slots, 1]
        normalized = x * torch.rsqrt(rms + self.eps)
        normalized = normalized * self.scale  # Apply per-dimension scaling

        if self.use_gate:
            gate_factor = torch.sigmoid(self.gate)  # Scalar gate (0-1)
            return normalized * gate_factor + x * (1 - gate_factor)
        else:
            return normalized

class MemoryLayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        use_gate: bool = True,
        init_scale: float = 1.0,
        init_gate: float = -4.0  # Start with gate closed (no normalization)
    ):
        super().__init__()
        self.use_gate = use_gate
        self.norm = nn.LayerNorm(dim)  # Normalizes across embedding dimensions per slot
        self.gate = nn.Parameter(torch.tensor([init_gate]))  # Scalar gate for this layer
        self.scale = nn.Parameter(torch.ones(dim) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)  # Apply LayerNorm across embedding dimensions (per slot)
        normalized = normalized * self.scale  # Per-dimension scaling

        if self.use_gate:
            gate_factor = torch.sigmoid(self.gate)  # Scalar gate (0-1)
            return normalized * gate_factor + x * (1 - gate_factor)
        else:
            return normalized

class MemoryNormConfig(TypedDict):
    num_slots: int
    decay: float
    use_scale: bool
    use_gate: bool
    init_gate: float
    init_scale: float
    per_dim_scale: bool

def init_memory_norm(
    norm_type: str,
    dim: int,
    num_slots: int = None,
    decay: float = 0.9,
    use_scale: bool = True,
    use_gate: bool = True,
    init_gate: float = -2.0,
    init_scale: float = 1.0,
    per_dim_scale: bool = False,
) -> nn.Module:
    assert norm_type in ['layer', 'rms', 'adaptive', 'positional', 'classic-rms']
    if norm_type == 'layer':
        return MemoryLayerNorm(dim, use_gate, init_scale, init_gate)
    elif norm_type == 'rms':
        return SimpleRMSMemoryNorm(dim, use_gate, init_scale, init_gate)
    elif norm_type == 'adaptive':
        return AdaptiveRMSMemoryNorm(dim, use_gate, decay, init_scale, init_gate)
    elif norm_type == 'positional':
        return AdaptivePositionalMemoryNorm(num_slots, dim, decay, use_scale, use_gate, init_gate, per_dim_scale)
    elif norm_type == 'classic-rms':
        return nn.RMSNorm(dim)
    return MemoryLayerNorm(dim, use_gate, init_scale, init_gate)
