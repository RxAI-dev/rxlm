import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union
import math
from .positional import RotaryPositionalEmbedding, RelativePositionalEmbedding
from .ff import get_activation_layer

class MultiHeadAttention(nn.Module):
    """Custom, extendable Multi-head attention layer, with RoPE support"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            rope_only_for_keys: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = True,
            is_causal: bool = False,
            use_bias: bool = False,
            use_output_bias: bool = True, # legacy compatibility - switch in model setting. Will be changed to False in next versions
            use_gated_attention: bool = False,
            gated_attention_activation: str = 'sigmoid',
            *args,
            **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.use_flash_attention = use_flash_attention
        self.is_causal = is_causal
        self.use_bias = use_bias
        self.use_output_bias = use_output_bias
        if use_relative_embeddings:
            self.use_flash_attention = False
            self.rel_embed = RelativePositionalEmbedding(max_seq_len, embed_dim // num_heads)
            self.rope = None
            self.rope_only_for_query = False
            self.rope_only_for_keys = False
        else:
            self.rel_embed = None
            self.rope = rope
            self.rope_only_for_query = rope_only_for_query
            self.rope_only_for_keys = rope_only_for_keys
        self.dropout = nn.Dropout(dropout)
        self._init_q(embed_dim)
        self._init_kv(embed_dim)
        self._init_out(embed_dim)
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.mask_cache: Optional[torch.Tensor] = None
        # from v0.3.34 - Gated Attention
        self.use_gated_attention = use_gated_attention
        if use_gated_attention:
            self.gated_attention_activation = gated_attention_activation
            self._init_gate(embed_dim)
        else:
            self.gated_attention_activation = None
            self.gate_proj = None
            self.gate_activation = None

    def _init_q(self, embed_dim: int):
        """Initialize query projection"""
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)

    def _init_kv(self, embed_dim: int):
        """Initialize key and value projections"""
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)

    def _init_out(self, embed_dim: int):
        """Initialize output projection"""
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_output_bias)

    def _init_gate(self, embed_dim: int):
        """Initialize gated attention"""
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=self.use_bias)
        self.gate_activation = get_activation_layer(self.gated_attention_activation)
        nn.init.normal_(self.gate_proj.weight, mean=1.0, std=0.01)

    def split_kv_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, -1, self.num_heads, d // self.num_heads).transpose(1, 2)

    def _split_q_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int, stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None):
        """Forward pass through query, key, and value projections, and split the results into heads"""
        q = self._split_q_head(self.q_proj(query), b, t, d)
        k = self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]
        v = self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]
        return q, k, v

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, separate: bool = False):
        if self.rope is not None:
            if self.rope_only_for_query:
                q = self.rope.forward_one(q)
            elif self.rope_only_for_keys:
                k = self.rope.forward_one(k)
            elif separate:
                q, k = self.rope.forward_one(q), self.rope.forward_one(k)
            else:
                q, k = self.rope(q, k)
        return q, k

    def _calculate_attn_weights(self, q: torch.Tensor, k: torch.Tensor, d: int, mask: torch.Tensor = None):
        """Calculate attention weights using scaled dot-product attention"""
        q, k = self._apply_rope(q, k)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d // self.num_heads) ** 0.5
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attn_logits, dim=-1)

    def _calculate_attn_weight_with_relative_embeddings(self, q: torch.Tensor, k: torch.Tensor,
                                                        mask: torch.Tensor = None):
        """Calculate attention weights using scaled dot-product attention and apply relative embedding"""
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        rel_pos_bias = self.rel_embed(q, k)
        attn_logits += rel_pos_bias
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attn_logits, dim=-1)

    def _transpose_output(self, attn_output: torch.Tensor, b: int, t: int, d: int):
        """Transpose attention output back to (B, T, D) shape"""
        return attn_output.transpose(1, 2).contiguous().view(b, t, d)

    def _calculate_output(self, attn_weights: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int):
        """Calculate the output by multiplying attention weights with values and concatenating heads"""
        return self._transpose_output(torch.matmul(attn_weights, v), b, t, d)

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int,
                         mask: torch.Tensor = None, enable_gqa: bool = False, generate_mode: bool = False):
        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask if not self.is_causal or generate_mode else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.is_causal if not generate_mode else False,
                enable_gqa=enable_gqa,
            )
        return self._transpose_output(attn_output, b, t, d)

    def _torch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int,
                         mask: torch.Tensor = None, enable_gqa: bool = False, generate_mode: bool = False):
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask if not self.is_causal or generate_mode else None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal if not generate_mode else False,
            enable_gqa=enable_gqa,
        )

        return self._transpose_output(attn_output, b, t, d)

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None, generate_mode: bool = False):
        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, generate_mode=generate_mode)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, generate_mode=generate_mode)

    def _calculate_attention_with_relative_embedding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None):
        attn_weights = self._calculate_attn_weight_with_relative_embeddings(q, k, mask=mask)
        attn_weights = self.dropout(attn_weights)
        return self._calculate_output(attn_weights, v, b, t, d)

    def reset_inner_cache(self):
        self.key_cache = None
        self.value_cache = None
        self.mask_cache = None

    def _forward_with_inner_cache(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
            current_positions: torch.Tensor = None,
    ):
        b, t, d = query.size()
        q = self._split_q_head(self.q_proj(query), b, t, d)

        if self.key_cache is None and self.value_cache is None:
            k = self.split_kv_head(self.k_proj(key), b, t, d)
            v = self.split_kv_head(self.v_proj(value), b, t, d)
            q, k = self._apply_rope(q, k)

            self.key_cache = torch.zeros(b, k.size(1), self.rope.max_seq_len, k.size(-1), device=k.device, dtype=k.dtype)
            self.value_cache = torch.zeros_like(self.key_cache).to(v.device, dtype=v.dtype)
            self.mask_cache = torch.zeros(b, 1, 1, self.rope.max_seq_len, device=mask.device, dtype=mask.dtype)

            self.key_cache[:, :, :t, :] = k * mask.squeeze(-2).unsqueeze(-1)
            self.value_cache[:, :, :t, :] = v * mask.squeeze(-2).unsqueeze(-1)
            self.mask_cache[:, :, :, :t] = mask
            attn_output = self._calculate_attention(q, k, v, b, t, d, mask=mask, generate_mode=False)
        else:
            new_k = self.split_kv_head(self.k_proj(key), b, t, d)
            new_v = self.split_kv_head(self.v_proj(value), b, t, d)
            q, new_k = self.rope.forward_on(q, new_k, current_positions)

            batch_range = torch.arange(b, device=q.device)
            self.key_cache[batch_range, :, current_positions, :] = new_k.squeeze(-2)
            self.value_cache[batch_range, :, current_positions, :] = new_v.squeeze(-2)
            self.mask_cache[batch_range, :, :, current_positions] = mask.squeeze(-1)

            k = self.key_cache
            v = self.value_cache
            mask = self.mask_cache
            attn_output = self._calculate_attention(q, k, v, b, t, d, mask=mask, generate_mode=True)

        if self.use_gated_attention:
            gate_logits = self.gate_proj(query)
            gate = self.gate_activation(gate_logits)
            attn_output = attn_output * gate

        return self.out_proj(attn_output)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
            stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
            use_self_attn_cache: bool = False,
            current_positions: torch.Tensor = None,
    ):
        if use_self_attn_cache:
            return self._forward_with_inner_cache(query, key, value, mask=mask, current_positions=current_positions)

        b, t, d = query.size()
        q, k, v = self._forward_qkv(query, key, value, b, t, d, stm_kv_cache=stm_kv_cache)
        if not self.rel_embed:
            if not stm_kv_cache or t != 1:
                q, k = self._apply_rope(q, k)
            elif stm_kv_cache and current_positions is not None:
                q = self.rope.forward_one_from(q, current_positions)
            attn_output = self._calculate_attention(q, k, v, b, t, d, mask=mask)
        else:
            attn_output = self._calculate_attention_with_relative_embedding(q, k, v, b, t, d, mask=mask)

        if self.use_gated_attention:
            gate_logits = self.gate_proj(query)
            gate = self.gate_activation(gate_logits)
            attn_output = attn_output * gate

        return self.out_proj(attn_output)


class GroupedQueryAttention(MultiHeadAttention):
    """Custom Grouped Query attention layer, with RoPE support"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_groups: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = False,
            is_causal: bool = False,
            use_bias: bool = False,
            *args,
            **kwargs,
    ):
        self.num_groups = num_groups
        super(GroupedQueryAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            *args,
            **kwargs,
        )
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

    def _init_kv(self, embed_dim: int):
        self.k_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)

    def split_kv_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, -1, self.num_groups, d // self.num_heads).transpose(1, 2)

    def _split_q_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int, stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None):
        """Override query, key, and value projections for GQA case - split data into heads and groups"""
        if not self.rel_embed:
            q = self._split_q_head(self.q_proj(query), b, t, d)
            k = self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]
            v = self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]
        else:
            # Relative embedding version is not working without this strange mapping - it will be removed in next versions
            group_heads = self.num_heads // self.num_groups
            head_dim = d // self.num_heads

            # Process Q
            q = self.q_proj(query).view(b, t, self.num_groups, group_heads, head_dim).permute(0, 2, 3, 1,
                                                                                              4)  # (B, G, group_heads, T, head_dim)
            # Process K and V
            k = self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]  # (B, G, S, head_dim)
            v = self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]  # (B, G, S, head_dim)

            # Expand and flatten to 4D tensors
            k = k.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)

            q = q.flatten(start_dim=1, end_dim=2)  # (B, H, T, head_dim)
            k = k.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
            v = v.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
        return q, k, v

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None, generate_mode: bool = False):
        is_gqa = self.num_heads != self.num_groups
        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=is_gqa, generate_mode=generate_mode)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=is_gqa, generate_mode=generate_mode)


class MultiQueryAttention(MultiHeadAttention):
    """Custom Multi Query attention layer, with RoPE support"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = False,
            is_causal: bool = False,
            use_bias: bool = False,
            *args,
            **kwargs,
    ):
        super(MultiQueryAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            *args,
            **kwargs
        )

    def _init_kv(self, embed_dim: int):
        """Override key/value initialization for MQA case"""
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.num_heads, bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.num_heads, bias=self.use_bias)

    def split_kv_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, -1, 1, d // self.num_heads).transpose(1, 2)

    def _split_q_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int, stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None):
        """Override query, key, and value projections for GQA case - use multiple heads
        for query and single for key/values"""
        if not self.rel_embed:
            q = self._split_q_head(self.q_proj(query), b, t, d)
            k = self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]
            v = self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]
        else:
            q = self._split_q_head(self.q_proj(query), b, t, d)
            k = (self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]).expand(-1, self.num_heads, -1, -1)
            v = (self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]).expand(-1, self.num_heads, -1, -1)
        return q, k, v

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None, generate_mode: bool = False):
        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=True, generate_mode=generate_mode)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=True, generate_mode=generate_mode)


class SparseQueryAttention(MultiHeadAttention):
    """Sparse Grouped Query attention layer, with RoPE support"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_groups: int,
            num_query_groups: int,
            dropout: float = 0.0,
            rope: RotaryPositionalEmbedding = None,
            rope_only_for_query: bool = False,
            use_relative_embeddings: bool = False,
            max_seq_len: int = 1024,
            use_flash_attention: bool = False,
            is_causal: bool = False,
            use_bias: bool = False,
            *args,
            **kwargs,
    ):
        self.num_groups = num_groups
        self.num_query_groups = num_query_groups
        super(SparseQueryAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            *args,
            **kwargs,
        )
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

    def _init_kv(self, embed_dim: int):
        self.k_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)

    def _init_q(self, embed_dim: int):
        self.q_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_query_groups), bias=self.use_bias)

    def _init_out(self, embed_dim: int):
        """Initialize output projection"""
        if self.num_query_groups < self.num_groups:
            # revSQA
            self.out_proj = nn.Linear(embed_dim // (self.num_heads // self.num_groups), embed_dim, bias=self.use_output_bias)
        else:
            self.out_proj = nn.Linear(embed_dim // (self.num_heads // self.num_query_groups), embed_dim, bias=self.use_output_bias)

    def _init_gate(self, embed_dim: int):
        if self.num_query_groups < self.num_groups:
            self.gate_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)
        else:
            self.gate_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_query_groups), bias=self.use_bias)
        nn.init.normal_(self.gate_proj.weight, mean=1.0, std=0.01)
        self.gate_activation = get_activation_layer(self.gated_attention_activation)

    def _transpose_output(self, attn_output: torch.Tensor, b: int, t: int, d: int):
        """Transpose attention output back to (B, T, D) shape"""
        if self.num_query_groups < self.num_groups:
            # revSQA
            return attn_output.transpose(1, 2).contiguous().view(b, t, d // (self.num_heads // self.num_groups))
        else:
            return attn_output.transpose(1, 2).contiguous().view(b, t, d // (self.num_heads // self.num_query_groups))

    def split_kv_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, -1, self.num_groups, d // self.num_heads).transpose(1, 2)

    def _split_q_head(self, projected: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        return projected.view(b, t, self.num_query_groups, d // self.num_heads).transpose(1, 2)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int, stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None):
        """Override query, key, and value projections for GQA case - split data into heads and groups"""
        if not self.rel_embed:
            q = self._split_q_head(self.q_proj(query), b, t, d)
            k = self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]
            v = self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]
        else:
            # Relative embedding version is not working without this strange mapping - it will be removed in next versions
            group_heads = self.num_heads // self.num_groups
            query_heads = self.num_heads // self.num_query_groups
            head_dim = d // self.num_heads
            # Process Q
            q = self._split_q_head(self.q_proj(query), b, t, d)  # (B, Q_G, T, head_dim)

            # Process K and V
            k = self.split_kv_head(self.k_proj(key), b, t, d) if stm_kv_cache is None else stm_kv_cache[0]  # (B, G, S, head_dim)
            v = self.split_kv_head(self.v_proj(value), b, t, d) if stm_kv_cache is None else stm_kv_cache[1]  # (B, G, S, head_dim)

            # Expand and flatten to 4D tensors
            q = q.unsqueeze(2).expand(-1, -1, query_heads, -1, -1)  # (B, Q_G, query_heads, T, head_dim)
            k = k.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)

            q = q.flatten(start_dim=1, end_dim=2)  # (B, Q, T, head_dim)
            k = k.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
            v = v.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
        return q, k, v

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, t: int, d: int, mask: torch.Tensor = None, generate_mode: bool = False):
        is_gqa = self.num_query_groups != self.num_groups and self.num_groups < self.num_query_groups

        # reversed SQA (revSQA)
        if self.num_query_groups < self.num_groups:
            q = q.repeat(1, self.num_groups // self.num_query_groups, 1, 1)  # Align q heads to kv heads

        if self.use_flash_attention:
            # Compute attention with FlashAttention
            return self._flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=is_gqa, generate_mode=generate_mode)
        else:
            # Compute attention using optimized PyTorch implementation
            return self._torch_attention(q.contiguous(), k.contiguous(), v.contiguous(), b, t, d, mask=mask, enable_gqa=is_gqa, generate_mode=generate_mode)


class LinearAttention(nn.Module):
    """
    Wrapper for flash-linear-attention layers (GLA, DeltaNet, Gated DeltaNet, KDA, MD-GDN).

    This provides a compatible interface with MultiHeadAttention for use in RxLM.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        linear_attn_type: Literal['gla', 'deltanet', 'gated_deltanet', 'kda', 'md_gdn'] = 'gla',
        mode: str = 'chunk',
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_gate: bool = True,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super(LinearAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear_attn_type = linear_attn_type

        # Import the appropriate linear attention layer
        if linear_attn_type == 'gla':
            from fla.layers import GatedLinearAttention
            self.attn_layer = GatedLinearAttention(
                mode=mode,
                hidden_size=embed_dim,
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_heads,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                use_output_gate=use_gate,
                norm_eps=norm_eps,
                layer_idx=layer_idx,
                **kwargs,
            )
        elif linear_attn_type == 'deltanet':
            from fla.layers import DeltaNet
            self.attn_layer = DeltaNet(
                mode=mode,
                hidden_size=embed_dim,
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_heads,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                use_gate=use_gate,
                norm_eps=norm_eps,
                layer_idx=layer_idx,
                **kwargs,
            )
        elif linear_attn_type == 'gated_deltanet':
            # Gated DeltaNet is essentially DeltaNet with use_gate=True
            from fla.layers import DeltaNet
            self.attn_layer = DeltaNet(
                mode=mode,
                hidden_size=embed_dim,
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_heads,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                use_gate=True,  # Always True for gated variant
                norm_eps=norm_eps,
                layer_idx=layer_idx,
                **kwargs,
            )
        elif linear_attn_type == 'kda':
            # Kimi Delta Attention - per-channel gating variant
            from fla.layers import KimiDeltaAttention
            head_dim = embed_dim // num_heads
            self.attn_layer = KimiDeltaAttention(
                hidden_size=embed_dim,
                expand_v=expand_v,
                head_dim=head_dim,
                num_heads=num_heads,
                mode=mode,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                norm_eps=norm_eps,
                layer_idx=layer_idx,
                **kwargs,
            )
        elif linear_attn_type == 'md_gdn':
            # Memory-Driven Gated DeltaNet - extends KDA with memory features
            from ..experimental.md_gdn import MemoryDrivenGatedDeltaNet
            head_dim = embed_dim // num_heads
            self.attn_layer = MemoryDrivenGatedDeltaNet(
                hidden_size=embed_dim,
                expand_v=expand_v,
                head_dim=head_dim,
                num_heads=num_heads,
                mode=mode,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                norm_eps=norm_eps,
                layer_idx=layer_idx,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown linear_attn_type: {linear_attn_type}")

        # For compatibility with existing code
        self.rope = None
        self.key_cache = None
        self.value_cache = None
        self.mask_cache = None

    def reset_inner_cache(self):
        """Reset cache for compatibility with MultiHeadAttention interface"""
        self.key_cache = None
        self.value_cache = None
        self.mask_cache = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        use_self_attn_cache: bool = False,
        current_positions: torch.Tensor = None,
        memory_state: torch.Tensor = None,
    ):
        """
        Forward pass compatible with MultiHeadAttention interface.

        For linear attention, query/key/value are expected to be the same (self-attention).
        The linear attention layers expect input of shape (batch, seq_len, hidden_size).

        Args:
            query: Query tensor (used as hidden_states for self-attention)
            key: Key tensor (ignored for self-attention)
            value: Value tensor (ignored for self-attention)
            mask: Attention mask
            stm_kv_cache: KV cache tuple (not used for linear attention)
            use_self_attn_cache: Whether to use caching
            current_positions: Current positions tensor (not used for linear attention)
            memory_state: STM state for MD-GDN [batch, memory_slots, hidden_size]
        """
        # Linear attention layers expect single input tensor for self-attention
        # In RxLM, for self-attention, query/key/value are all the same
        hidden_states = query

        # Prepare attention mask if provided
        # flash-linear-attention expects mask of shape (batch, seq_len) or None
        attention_mask = None
        if mask is not None:
            # Convert from (batch, 1, seq_len, seq_len) to (batch, seq_len)
            if mask.dim() == 4:
                # Take the diagonal or last row (for causal masks)
                attention_mask = mask[:, 0, -1, :]
            elif mask.dim() == 3:
                attention_mask = mask[:, -1, :]
            elif mask.dim() == 2:
                attention_mask = mask

        # Handle caching for inference
        past_key_values = None
        use_cache = use_self_attn_cache

        # Call the linear attention layer
        # MD-GDN requires memory_state
        if self.linear_attn_type == 'md_gdn':
            if memory_state is None:
                raise ValueError("MD-GDN requires memory_state to be provided")
            output, _ = self.attn_layer(
                hidden_states=hidden_states,
                memory_state=memory_state,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
        else:
            # Standard linear attention (GLA, DeltaNet, KDA)
            output, _, past_key_values = self.attn_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        return output


def init_attention(
        embed_dim: int,
        num_heads: int,
        attention_type: str,
        gqa_groups: int = 1,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding = None,
        rope_only_for_query: bool = False,
        rope_only_for_keys: bool = False,
        use_relative_embeddings: bool = False,
        max_seq_len: int = 1024,
        use_flash_attention: bool = False,
        is_causal: bool = False,
        use_bias: bool = False,
        num_query_groups: int = 1,
        is_linear_attention: bool = False,
        linear_attn_mode: str = 'chunk',
        linear_attn_expand_k: float = None,
        linear_attn_expand_v: float = None,
        linear_attn_use_short_conv: bool = False,
        linear_attn_conv_size: int = 4,
        linear_attn_use_gate: bool = True,
        linear_attn_layer_idx: int = None,
        linear_attn_norm_eps: float = 1e-5,
        use_gated_attention: bool = False,
        gated_attention_activation: str = 'sigmoid',
        use_output_bias: bool = True, # legacy compat
) -> Union[MultiHeadAttention, LinearAttention]:
    if not is_linear_attention:
        assert attention_type in ['mha', 'gqa', 'mqa', 'sqa'], "Error, attention type should be one of: 'mha', 'gqa', 'mqa' or 'sqa'"

        if attention_type == 'sqa':
            return SparseQueryAttention(
                embed_dim,
                num_heads,
                gqa_groups,
                num_query_groups,
                dropout=dropout,
                rope=rope,
                use_relative_embeddings=use_relative_embeddings,
                max_seq_len=max_seq_len,
                rope_only_for_query=rope_only_for_query,
                rope_only_for_keys=rope_only_for_keys,
                use_flash_attention=use_flash_attention,
                is_causal=is_causal,
                use_bias=use_bias,
                use_gated_attention=use_gated_attention,
                gated_attention_activation=gated_attention_activation,
                use_output_bias=use_output_bias,
            )
        elif attention_type == "gqa":
            return GroupedQueryAttention(
                embed_dim,
                num_heads,
                gqa_groups,
                dropout=dropout,
                rope=rope,
                use_relative_embeddings=use_relative_embeddings,
                max_seq_len=max_seq_len,
                rope_only_for_query=rope_only_for_query,
                rope_only_for_keys=rope_only_for_keys,
                use_flash_attention=use_flash_attention,
                is_causal=is_causal,
                use_bias=use_bias,
                use_gated_attention=use_gated_attention,
                gated_attention_activation=gated_attention_activation,
                use_output_bias=use_output_bias
            )
        elif attention_type == "mqa":
            return MultiQueryAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                rope=rope,
                use_relative_embeddings=use_relative_embeddings,
                max_seq_len=max_seq_len,
                rope_only_for_query=rope_only_for_query,
                rope_only_for_keys=rope_only_for_keys,
                use_flash_attention=use_flash_attention,
                is_causal=is_causal,
                use_bias=use_bias,
                use_gated_attention=use_gated_attention,
                gated_attention_activation=gated_attention_activation,
                use_output_bias=use_output_bias
            )
        else:
            return MultiHeadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                rope=rope,
                use_relative_embeddings=use_relative_embeddings,
                max_seq_len=max_seq_len,
                rope_only_for_query=rope_only_for_query,
                rope_only_for_keys=rope_only_for_keys,
                use_flash_attention=use_flash_attention,
                is_causal=is_causal,
                use_bias=use_bias,
                use_gated_attention=use_gated_attention,
                gated_attention_activation=gated_attention_activation,
                use_output_bias=use_output_bias
            )
    else:
        return LinearAttention(
            embed_dim,
            num_heads,
            linear_attn_type=attention_type,
            mode=linear_attn_mode,
            expand_k=linear_attn_expand_k,
            expand_v=linear_attn_expand_v,
            use_short_conv=linear_attn_use_short_conv,
            conv_size=linear_attn_conv_size,
            use_gate=linear_attn_use_gate,
            layer_idx=linear_attn_layer_idx,
            norm_eps=linear_attn_norm_eps,
        )
