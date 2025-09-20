import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformers.attention import MultiHeadAttention, GroupedQueryAttention
from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.moe import MoeRouter

# Created by Reactive AI

class GroupedMoeAttention(GroupedQueryAttention):
    """
    Vectorized implementation calculates all expert heads for each token and selecting active tokens later. Linear layers
    for Attention are rather small, compared to MoE Feed Forward layers, so it's possible that it will be faster than filtering
    experts - it has to be tested.

    Grouped MoE Attention (GMA) - GQA extended with Mixture-of-Experts (MoE) routing.

    Instead of mapping keys/values to static head groups, it dynamically selects head expert groups. It has the same
    number of total keys/values heads as query heads, but uses only a selected group for attention calculation.
    - with num_groups set to 1, it will be MoE MultiQueryAttention

    Compared to traditional GQA/MQA, it should provide better performance, because lot less data could be lost using
    this approach - we are training the full number of keys/values heads, while using only a group.

    In case of efficiency, it should be close to GQA/MQA linear performance, but with a small MoE routing overhead.

    Optionally, it could use even more expert heads than attention heads - in example:
    - 512 dim divided into 16 heads with 32 dim, using 4 head groups - may use i.e., 24 total expert heads - still only
    4 will be used for attention calculation, while 16 is used to split dimensions (in that case it will have 16 query heads)

    © 2025 Adam Filipek
    """

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
            num_experts: int = None,
            *args,
            **kwargs,
    ):
        self.num_experts = num_experts if num_experts is not None else num_heads
        super(GroupedMoeAttention, self).__init__(
            embed_dim,
            num_heads,
            num_groups=num_groups,
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

    def router_loss(self):
        return self.router.aux_loss

    def _init_kv(self, embed_dim: int):
        self.router = MoeRouter(embed_dim, self.num_experts, top_k=self.num_groups)
        hidden_dim = embed_dim // self.num_heads
        moe_dim = hidden_dim * self.num_experts
        self.k_proj = nn.Linear(embed_dim, moe_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, moe_dim, bias=self.use_bias)
        # self.wk = nn.Parameter(torch.empty(self.num_experts, embed_dim, hidden_dim))
        # self.bk = nn.Parameter(torch.zeros(self.num_experts, hidden_dim)) if self.use_bias else None
        # self.wv = nn.Parameter(torch.empty(self.num_experts, embed_dim, hidden_dim))
        # self.bv = nn.Parameter(torch.zeros(self.num_experts, hidden_dim)) if self.use_bias else None
        # self._init_experts()

    def _init_experts(self):
        pass
        # torch.nn.init.xavier_uniform_(self.wk)
        # torch.nn.init.xavier_uniform_(self.wv)
        # if self.use_bias:
        #     torch.nn.init.zeros_(self.bk)
        #     torch.nn.init.zeros_(self.bv)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int,
                     stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None, skip_query_processing: bool = False):
        # Process Query as in GQA
        q = self.q_proj(query).view(b, t, self.num_heads, -1).transpose(1, 2) if not skip_query_processing else query

        # Key/Value MoE routing
        B, S, D = key.shape
        key_flat = key.reshape(-1, D)
        weights, indices = self.router(key_flat)  # (B*S, num_groups), (B*S, num_groups)
        weights = weights.view(B, S, self.num_groups, 1)
        indices = indices.view(B, S, self.num_groups)

        # Compute all experts' projections
        k_all = self.k_proj(key_flat).view(B, S, self.num_experts, -1) # [B, S, num_experts, head_dim]
        v_all = self.v_proj(value).view(B, S, self.num_experts, -1) # [B, S, num_experts, head_dim]

        # Gather top-k experts using expanded indices
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, k_all.size(-1)) # [B, num_groups, S, head_dim]
        selected_k = torch.gather(k_all, 2, expanded_indices) # [B, num_groups, S, head_dim]
        selected_v = torch.gather(v_all, 2, expanded_indices) # [B, num_groups, S, head_dim]

        # Weighted
        weighted_k = (selected_k * weights).to(selected_k.device, dtype=selected_k.dtype) # [B, S, num_groups, head_dim]
        weighted_v = (selected_v * weights).to(selected_k.device, dtype=selected_k.dtype)  # [B, S, num_groups, head_dim]

        # Reshape to GQA format
        k = weighted_k.view(B, S, self.num_groups, -1).permute(0, 2, 1, 3) # [B, num_groups, S, head_dim]
        v = weighted_v.view(B, S, self.num_groups, -1).permute(0, 2, 1, 3) # [B, num_groups, S, head_dim]

        if self.rel_embed:
            group_heads = self.num_heads // self.num_groups

            k = k.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_heads, -1, -1)  # (B, G, group_heads, S, head_dim)

            k = k.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)
            v = v.flatten(start_dim=1, end_dim=2)  # (B, H, S, head_dim)

        return q, k, v


class DeepMoeAttention(GroupedMoeAttention):
    """
    Deep MoE Attention (SMA) - Grouped MoE Attention extended even more for sublinear computational efficiency.

    In addition to using Mixture-of-Experts (MoE) for key/value head groups, SMA is also using dynamically selected
    query heads - with that approach, each token could attend to every other token, but only partially - only some part of
    information from each token is used to identify related information parts from other tokens. So, DMA is not spatially
    sparse (has access to all tokens), but rather structurally sparse (has access only to the part of token's information).

    This solution could reduce the computational complexity of attention operation to sublinear level (<O(N)) and provide
    a viable and efficient alternative to spatial sparse attention mechanisms like Flex Attention.

    © 2025 Adam Filipek
    """

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
            num_experts: int = None,
            num_query_experts: int = None,
            num_query_groups: int = None,
            *args,
            **kwargs,
    ):
        self.num_query_experts = num_query_experts if num_query_experts is not None else num_heads
        self.num_query_groups = num_query_groups if num_query_groups is not None else num_groups
        super(DeepMoeAttention, self).__init__(
            embed_dim,
            num_heads,
            num_groups=num_groups,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            num_experts=num_experts,
            *args,
            **kwargs,
        )

    def _init_q(self, embed_dim: int):
        self.query_router = MoeRouter(embed_dim, self.num_query_experts, top_k=self.num_query_groups)
        hidden_dim = embed_dim // self.num_heads
        moe_dim = hidden_dim * self.num_query_experts
        self.q_proj = nn.Linear(embed_dim, moe_dim)
        # self.wq = nn.Parameter(torch.empty(self.num_query_experts, embed_dim, hidden_dim))
        # self.bq = nn.Parameter(torch.zeros(self.num_query_experts, hidden_dim)) if self.use_bias else None
        # self._init_query_experts()

    def _init_query_experts(self):
        pass
        # torch.nn.init.xavier_uniform_(self.wq)
        # if self.use_bias:
        #     torch.nn.init.zeros_(self.bq)

    def _init_out(self, embed_dim: int):
        """Initialize output projection"""
        out_hidden_dim = embed_dim // self.num_heads * self.num_query_groups
        self.out_proj = nn.Linear(out_hidden_dim, embed_dim)

    def _transpose_output(self, attn_output: torch.Tensor, b: int, t: int, d: int):
        """Transpose attention output back to (B, T, D) shape"""
        out_hidden_dim = d // self.num_heads * self.num_query_groups
        return attn_output.transpose(1, 2).contiguous().view(b, t, out_hidden_dim)

    def _forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, b: int, t: int, d: int,
                     stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None, skip_query_processing: bool = False):
        B, T, D = query.shape
        query_flat = query.reshape(-1, D)
        weights_q, indices_q = self.query_router(query_flat)
        weights_q = weights_q.view(B, T, self.num_query_groups, 1)
        indices_q = indices_q.view(B, T, self.num_query_groups)

        q_all = self.q_proj(query_flat).view(B, T, self.num_query_experts, -1) # [B, num_groups, S, head_dim]

        # Gather top-k experts
        expanded_indices = indices_q.unsqueeze(-1).expand(-1, -1, -1, q_all.size(-1)) # [B, T, num_query_groups, head_dim]
        selected_q = torch.gather(q_all, 2, expanded_indices)  # [B, T, num_query_groups, head_dim]

        # Weighted sum
        q = (selected_q * weights_q).to(selected_q.device, dtype=selected_q.dtype)  # [B, T, num_query_groups, head_dim]
        q = q.view(B, T, self.num_query_groups, -1).permute(0, 2, 1, 3)  # [B, num_query_groups, T, head_dim]

        return super()._forward_qkv(q, key, value, b, t, d, stm_kv_cache=stm_kv_cache, skip_query_processing=True)


# Others
class FlexAttention(MultiHeadAttention):
    """Flex attention layer, with RoPE support"""

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
        num_global_tokens: int = 16,
        window_size: int = 128,
    ):
        super(FlexAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
        )
        self.head_dim = embed_dim // num_heads
        self.num_global_tokens = num_global_tokens
        self.window_size = window_size

        # Learnable global tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, embed_dim))

    def _split_heads(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        b, t, d = x.size()
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.size()
        return self._transpose_output(x, b, t, h * d)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
                                       is_causal=self.is_causal)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        b, t, d = query.size()

        # Prepend global tokens
        global_tokens = self.global_tokens.expand(b, -1, -1)
        x = torch.cat([global_tokens, query], dim=1)

        # Project Q, K, V
        q = self._split_heads(self.q_proj(x), is_query=True)
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # Apply RoPE
        if self.rope:
            q, k = self._apply_rope(q, k, separate=True)

        # Split Q into global and local
        global_q = q[:, :, :self.num_global_tokens]  # (B, H, G, D)
        local_q = q[:, :, self.num_global_tokens:]   # (B, H, L, D)

        # Global attention
        global_attn = self._sdpa(global_q, k, v, mask=mask)

        # Local attention with windowed slicing (no large masks)
        L = local_q.size(2)
        S = k.size(2)
        windowed_attn = []

        for i in range(0, L, self.window_size):
            start = i
            end = min(i + self.window_size, L)
            window_q = local_q[:, :, start:end]  # (B, H, W, D)

            # Use only relevant keys/values (same window)
            k_window_start = max(0, start - self.window_size)
            k_window_end = min(S, end + self.window_size)
            window_k = k[:, :, k_window_start:k_window_end]
            window_v = v[:, :, k_window_start:k_window_end]

            window_attn = self._sdpa(window_q, window_k, window_v, mask=None)

            windowed_attn.append(window_attn)

        # Concat local attention
        local_attn = torch.cat(windowed_attn, dim=2)

        # Combine global and local
        attn = torch.cat([global_attn, local_attn], dim=2)
        output = self._merge_heads(attn)
        output = self.out_proj(output)

        return output[:, self.num_global_tokens:, :]


class FlexSparseQueryAttention(FlexAttention):
    """Combined Flex and Sparse Query attention layer, with RoPE support"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_groups: int,
        num_query_groups: int,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding = None,
        rope_only_for_query: bool = False,
        rope_only_for_keys: bool = False,
        use_relative_embeddings: bool = False,
        max_seq_len: int = 1024,
        use_flash_attention: bool = True,
        is_causal: bool = False,
        use_bias: bool = False,
        num_global_tokens: int = 16,
        window_size: int = 128,
    ):
        self.num_groups = num_groups
        self.num_query_groups = num_query_groups
        super(FlexSparseQueryAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_relative_embeddings=use_relative_embeddings,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            num_global_tokens=num_global_tokens,
            window_size=window_size,
        )
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

    def _init_kv(self, embed_dim: int):
        self.k_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_groups), bias=self.use_bias)

    def _init_q(self, embed_dim: int):
        self.q_proj = nn.Linear(embed_dim, embed_dim // (self.num_heads // self.num_query_groups), bias=self.use_bias)

    def _init_out(self, embed_dim: int):
        """Initialize output projection"""
        self.out_proj = nn.Linear(embed_dim // (self.num_heads // self.num_query_groups), embed_dim)

    def _split_heads(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        b, t, d = x.size()
        return x.view(b, t, self.num_query_groups if is_query else self.num_groups, self.head_dim).transpose(1, 2)

    def _transpose_output(self, attn_output: torch.Tensor, b: int, t: int, d: int):
        """Transpose attention output back to (B, T, D) shape"""
        return attn_output.transpose(1, 2).contiguous().view(b, t, d // (self.num_heads // self.num_query_groups))

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.size()
        return self._transpose_output(x, b, t, self.embed_dim)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        is_gqa = self.num_query_groups != self.num_groups
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
                                       is_causal=self.is_causal, enable_gqa=is_gqa)


class InfiniteAttention(MultiHeadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            kernel_size: int = 128,
            use_rotary: bool = True,
            **kwargs
    ):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.kernel_size = kernel_size
        self.use_rotary = use_rotary
        self.register_buffer("fourier_basis", self._init_fourier_basis(embed_dim))

    def _init_fourier_basis(self, embed_dim):
        # Initialize Fourier features for positional encoding
        freqs = torch.randn(embed_dim // 2)
        return freqs

    def _positional_encodings(self, x: torch.Tensor, device: torch.device):
        """Generate positional encodings for arbitrary sequence length."""
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=device).float()
        fourier_features = torch.einsum("d, s -> sd", self.fourier_basis, pos)
        pe = torch.cat([torch.sin(fourier_features), torch.cos(fourier_features)], dim=1)
        return pe.unsqueeze(0).expand(x.size(0), -1, -1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        b, t, d = query.size()
        # Add positional encodings
        pe = self._positional_encodings(query, query.device)
        query = query + pe
        key = key + pe

        # Split into chunks for kernel-based attention
        chunks = []
        for i in range(0, t, self.kernel_size):
            chunk = query[:, i:i + self.kernel_size]
            chunks.append(chunk)

        # Compute attention for each chunk
        attn_output = []
        for chunk in chunks:
            q, k, v = self._forward_qkv(chunk, key, value, b, chunk.size(1), d)
            # Use kernel approximation (e.g., Performer)
            attn = self._performer_attention(q, k, v)
            attn_output.append(attn)

        # Concatenate and apply output projection
        output = torch.cat(attn_output, dim=1)
        return self.out_proj(output)

    def _performer_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Performer kernel approximation (simplified)
        # TODO: Replace with preferred kernel method
        q = q / (q.shape[-1] ** 0.5)
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = torch.softmax(attn, dim=-1)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)

def init_experimental_attention(
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
        num_experts: int = None,
        num_query_experts: int = None,
        num_query_groups: int = None,
        num_global_tokens: int = 16,
        window_size: int = 128,
) -> MultiHeadAttention:
    assert attention_type in ['gma', 'dma', 'flex', 'flex-sqa'], "Error, attention type should be one of: 'gma', 'dma', 'flex', 'flex-sqa"

    if attention_type == "gma":
        return GroupedMoeAttention(
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
            num_experts=num_experts,
        )
    elif attention_type == "dma":
        return DeepMoeAttention(
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
            num_experts=num_experts,
            num_query_experts=num_query_experts,
            num_query_groups=num_query_groups,
        )
    elif attention_type == "flex":
        return FlexAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            max_seq_len=max_seq_len,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            num_global_tokens=num_global_tokens,
            window_size=window_size,
        )
    else:
        return FlexSparseQueryAttention(
            embed_dim,
            num_heads,
            gqa_groups,
            num_query_groups,
            dropout=dropout,
            rope=rope,
            max_seq_len=max_seq_len,
            rope_only_for_query=rope_only_for_query,
            rope_only_for_keys=rope_only_for_keys,
            use_flash_attention=use_flash_attention,
            is_causal=is_causal,
            use_bias=use_bias,
            num_global_tokens=num_global_tokens,
            window_size=window_size,
        )

