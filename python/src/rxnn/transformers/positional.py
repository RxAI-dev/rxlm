import torch
from torch import nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) layer - recommended for positional encoding"""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000, *args, **kwargs):
        super(RotaryPositionalEmbedding, self).__init__(*args, **kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq) # must stay for models compatibility
        # Pre-cache freqs for max_len
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        self.register_buffer('cache', freqs)

    def update_max_len(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        self.cache = freqs

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        # Prepare RoPE Frequencies
        freqs = self._prepare_freqs(seq_len).to(dtype=q.dtype)

        # Apply the rotation to the queries
        q_embed = self._rotate(q, freqs)
        # Apply the rotation to the keys
        k_embed = self._rotate(k, freqs)

        return q_embed, k_embed

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        # Prepare RoPE Frequencies
        freqs = self._prepare_freqs(seq_len).to(dtype=x.dtype)

        # Apply the rotation to the sequence
        x_embed = self._rotate(x, freqs)

        return x_embed

    def forward_one_from(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cached_freqs = self._prepare_freqs(self.max_seq_len).to(dtype=x.dtype)
        freqs = cached_freqs.expand(batch_size, cached_freqs.size(1), cached_freqs.size(2), cached_freqs.size(3))
        freqs = freqs[torch.arange(batch_size, device=x.device), :, pos, :]
        freqs = freqs.unsqueeze(-2)

        # Apply the rotation to the sequence
        x_embed = self._rotate(x, freqs)

        return x_embed

    def forward_on(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)
        cached_freqs = self._prepare_freqs(self.max_seq_len).to(dtype=q.dtype)
        freqs = cached_freqs.expand(batch_size, cached_freqs.size(1), cached_freqs.size(2), cached_freqs.size(3))
        freqs = freqs[torch.arange(batch_size, device=q.device), :, pos, :]
        freqs = freqs.unsqueeze(-2)

        # Apply the rotation to the sequence
        q_embed = self._rotate(q, freqs)
        k_embed = self._rotate(k, freqs)

        return q_embed, k_embed

    def _prepare_freqs(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            self.update_max_len(seq_len)
        return self.cache[None, None, :seq_len, :]

    def _rotate(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # Apply the rotation
        x_rotated1 = x1 * torch.cos(freqs) - x2 * torch.sin(freqs)
        x_rotated2 = x1 * torch.sin(freqs) + x2 * torch.cos(freqs)
        # Concatenate the rotated parts back together
        x_rotated = torch.cat((x_rotated1, x_rotated2), dim=-1)
        return x_rotated


class AbsolutePositionalEmbedding(nn.Module):
    """Absolute Positional Embedding layer (legacy) - not recommended for memory-augmented Reactive Transformers"""

    def __init__(self, max_seq_len: int, embed_dim: int, *args, **kwargs):
        super(AbsolutePositionalEmbedding, self).__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create position indices
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        # Get position embeddings
        pos_embeddings = self.position_embeddings(positions)
        # Add position embeddings to the input embeddings
        return x + pos_embeddings


class RelativePositionalEmbedding(nn.Module):
    """Relative Positional Embedding layer (legacy) - not compatible with Flash Attention and not recommended for positional encoding"""

    def __init__(self, max_seq_len: int, embed_dim: int, *args, **kwargs):
        super(RelativePositionalEmbedding, self).__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.position_embeddings = nn.Embedding(2 * max_seq_len - 1, embed_dim)
        self.embed_dim_sqrt = math.sqrt(embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_len = q.size(2)
        k_len = k.size(2)

        # Create relative position indices
        indices = torch.arange(q_len, device=q.device)[:, None] - torch.arange(k_len, device=k.device)[None, :]
        indices += self.max_seq_len - 1  # Shift to non-negative
        indices = torch.clamp(indices, 0, 2 * self.max_seq_len - 2)

        # Get embeddings
        rel_emb = self.position_embeddings(indices)

        rel_emb = rel_emb.permute(2, 0, 1)
        rel_pos_bias = torch.einsum('bhqd, dqk -> bhqk', q, rel_emb)
        return rel_pos_bias / self.embed_dim_sqrt
