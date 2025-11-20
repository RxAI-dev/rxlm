# -*- coding: utf-8 -*-
# Copyright (c) 2025, Reactive AI
# Memory-Driven Gated DeltaNet (MD-GDN)
# Extension of Kimi Delta Attention with memory-based features for RxLM

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MemoryDrivenGatedDeltaNet(nn.Module):
    """
    Memory-Driven Gated DeltaNet (MD-GDN) - Extended Kimi Delta Attention

    This layer extends KDA with two key features for integration with RxLM's memory system:
    1. Continuous State with STM Correction: Initializes linear attention recurrent state
       from a gated combination of previous recurrent state and compressed STM state
    2. Memory-Conditioned Gating: Conditions alpha/beta gates on both the current
       sequence and the memory state

    Args:
        hidden_size: The hidden size of the input. Default: 2048.
        expand_v: The expansion ratio for the value dimension. Default: 1.0.
        head_dim: The dimension of each head. Default: 128.
        num_heads: The number of heads. Default: 16.
        num_v_heads: The number of heads for value projection. Default: None.
        mode: Which kernel to use. Currently supports 'chunk' mode. Default: 'chunk'.
        use_short_conv: Whether to use short convolutions. Default: False.
        conv_size: Kernel size of short convolution. Default: 4.
        conv_bias: Whether to use bias in short convolution. Default: False.
        layer_idx: The index of the layer. Default: None.
        norm_eps: The epsilon value for normalization. Default: 1e-5.
        memory_fusion_mode: How to fuse memory with gates. Options: 'concat', 'add'. Default: 'add'.
        use_stm_correction: Whether to use STM state correction. Default: True.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: Optional[int] = None,
        mode: str = 'chunk',
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        memory_fusion_mode: str = 'add',
        use_stm_correction: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.layer_idx = layer_idx
        self.memory_fusion_mode = memory_fusion_mode
        self.use_stm_correction = use_stm_correction

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        # Consistency checks
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v,
                           self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value"
            )

        assert mode == 'chunk', f"Only 'chunk' mode is currently supported, got {mode}"
        assert memory_fusion_mode in ['concat', 'add'], \
            f"memory_fusion_mode must be 'concat' or 'add', got {memory_fusion_mode}"

        # Standard KDA projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Optional short convolutions
        if use_short_conv:
            from fla.modules import ShortConvolution
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        # Gate projections for delta rule
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Memory-conditioned gate projections
        if memory_fusion_mode == 'concat':
            # Concatenation mode: project concatenated features
            self.mem_f_proj = nn.Linear(hidden_size * 2, self.key_dim, bias=False)
            self.mem_b_proj = nn.Linear(hidden_size * 2, self.num_heads, bias=False)
        else:  # 'add'
            # Additive mode: separate projections for memory
            self.mem_f_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.mem_b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Learnable parameters for gating
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        # STM state correction components
        if use_stm_correction:
            state_dim = self.num_heads * self.head_k_dim * self.head_v_dim
            # Project STM to recurrent state space
            self.stm_to_state = nn.Linear(hidden_size, state_dim, bias=False)
            # Gate for interpolating between persistent and STM-derived state
            self.state_gate = nn.Linear(hidden_size * 2, state_dim, bias=True)

        # Output projection
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Persistent recurrent state (for continuous state mechanism)
        self.persistent_state: Optional[torch.Tensor] = None

    def compute_memory_conditioned_gates(
        self,
        x: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute memory-conditioned gates (g and beta).

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            memory_state: STM state [batch, memory_slots, hidden_size]

        Returns:
            g: Gating tensor [batch, seq_len, key_dim]
            beta: Beta parameter [batch, seq_len, num_heads]
        """
        # Compress memory state via mean pooling
        mem_context = memory_state.mean(dim=1, keepdim=True)  # [batch, 1, hidden_size]

        # Expand to match sequence length
        mem_context = mem_context.expand(-1, x.size(1), -1)  # [batch, seq_len, hidden_size]

        if self.memory_fusion_mode == 'concat':
            # Concatenate sequence and memory context
            combined = torch.cat([x, mem_context], dim=-1)  # [batch, seq_len, 2*hidden_size]
            g_base = self.mem_f_proj(combined)
            beta_base = self.mem_b_proj(combined)
        else:  # 'add'
            # Additive fusion
            g_seq = self.f_proj(x)
            g_mem = self.mem_f_proj(mem_context)
            g_base = g_seq + g_mem

            beta_seq = self.b_proj(x)
            beta_mem = self.mem_b_proj(mem_context)
            beta_base = beta_seq + beta_mem

        # Apply gating logic (similar to KDA's fused_kda_gate)
        g = F.silu(g_base + self.dt_bias)
        g = g * F.softplus(self.A_log.unsqueeze(0).unsqueeze(0))
        beta = F.sigmoid(beta_base)

        return g, beta

    def compute_stm_corrected_state(
        self,
        memory_state: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute STM-corrected initial state for linear attention.

        Args:
            memory_state: STM state [batch, memory_slots, hidden_size]
            batch_size: Batch size

        Returns:
            corrected_state: Initial recurrent state [batch, num_heads, head_k_dim, head_v_dim]
        """
        if not self.use_stm_correction:
            return None

        # Compress STM state
        stm_compressed = memory_state.mean(dim=1)  # [batch, hidden_size]

        # Project to state space
        stm_as_state = self.stm_to_state(stm_compressed)  # [batch, state_dim]
        stm_as_state = stm_as_state.view(
            batch_size, self.num_heads, self.head_k_dim, self.head_v_dim
        )

        if self.persistent_state is None:
            # First interaction - use STM-derived state
            return stm_as_state
        else:
            # Combine persistent state with STM correction
            # Flatten persistent state for gating
            persistent_flat = self.persistent_state.flatten(-2)  # [batch, num_heads, head_k_dim*head_v_dim]
            persistent_flat = persistent_flat.flatten(1)  # [batch, num_heads*head_k_dim*head_v_dim]

            # Compute gate
            gate_input = torch.cat([persistent_flat, stm_compressed], dim=-1)
            gate = torch.sigmoid(self.state_gate(gate_input))
            gate = gate.view(batch_size, self.num_heads, self.head_k_dim, self.head_v_dim)

            # Interpolate between persistent and STM-derived state
            corrected_state = gate * self.persistent_state + (1 - gate) * stm_as_state

            return corrected_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for MD-GDN.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            memory_state: STM state [batch, memory_slots, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]
            use_cache: Whether to cache the recurrent state

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            new_state: Updated recurrent state (if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        if self.use_short_conv:
            # This would require FLA's ShortConvolution module
            # For now, we use simpler approach
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # Memory-conditioned gates
        g, beta = self.compute_memory_conditioned_gates(hidden_states, memory_state)

        # Reshape to heads
        q = rearrange(q, 'b l (h d) -> b l h d', d=self.head_k_dim)
        k = rearrange(k, 'b l (h d) -> b l h d', d=self.head_k_dim)
        v = rearrange(v, 'b l (h d) -> b l h d', d=self.head_v_dim)
        g = rearrange(g, 'b l (h d) -> b l h d', d=self.head_k_dim)

        # STM-corrected initial state
        initial_state = self.compute_stm_corrected_state(memory_state, batch_size)

        # Apply chunked linear attention with delta rule
        # Simplified implementation - in production, use optimized kernels
        o, new_state = self._chunk_linear_attention(
            q, k, v, g, beta, initial_state, attention_mask
        )

        # Update persistent state
        if use_cache and new_state is not None:
            self.persistent_state = new_state.detach()

        # Output normalization and projection
        o_gated = self.g_proj(hidden_states)
        o_gated = rearrange(o_gated, 'b l (h d) -> b l h d', d=self.head_v_dim)

        # Apply gated normalization
        o = self.o_norm(o) * F.sigmoid(o_gated)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        return o, new_state

    def _chunk_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified chunked linear attention with delta rule.

        This is a reference implementation. For production, use optimized kernels
        from flash-linear-attention library.
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        # L2 normalize q and k
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Initialize state
        if initial_state is None:
            h = torch.zeros(
                batch_size, num_heads, self.head_k_dim, self.head_v_dim,
                device=q.device, dtype=q.dtype
            )
        else:
            h = initial_state

        # Chunk size for processing
        chunk_size = 64
        outputs = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            q_chunk = q[:, start:end]  # [batch, chunk, heads, dim]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            g_chunk = g[:, start:end]
            beta_chunk = beta[:, start:end]

            # Process chunk
            o_chunk, h = self._process_chunk(
                q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, h
            )
            outputs.append(o_chunk)

        output = torch.cat(outputs, dim=1)
        return output, h

    def _process_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single chunk with delta rule.

        Simplified implementation of gated delta rule for linear attention.
        """
        batch_size, chunk_len, num_heads, _ = q.shape

        # Expand beta to match dimensions
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # [batch, chunk, heads, 1, 1]
        g = g.unsqueeze(-1)  # [batch, chunk, heads, dim, 1]

        outputs = []

        for t in range(chunk_len):
            # Query at time t
            q_t = q[:, t:t+1]  # [batch, 1, heads, dim]
            k_t = k[:, t:t+1]
            v_t = v[:, t:t+1]
            g_t = g[:, t:t+1]
            beta_t = beta[:, t:t+1]

            # Output: o_t = q_t @ h
            o_t = torch.einsum('bhqd,bhdk->bhqk', q_t, h)
            outputs.append(o_t)

            # State update with delta rule: h = beta * h + g * (k^T @ v)
            kv = torch.einsum('bhkd,bhkv->bhdv', k_t, v_t)
            h = beta_t * h + g_t * kv

        output = torch.cat(outputs, dim=1)
        return output, h

    def reset_state(self):
        """Reset the persistent recurrent state."""
        self.persistent_state = None
