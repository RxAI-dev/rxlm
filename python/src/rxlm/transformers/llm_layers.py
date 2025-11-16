import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ff import FeedForward, GatedFeedForward
from .moe import MoeFeedForward, GatedMoeFeedForward

class ClassicTransformerLayer(nn.Module):
    """Classic Transformer layer - classic decoder-only/encoder-only Transformer layer with self-attention and Feed-Forward network."""

    def __init__(
            self,
            embed_dim: int,
            ff_dim: int,
            self_attention: MultiHeadAttention,
            use_rms_norm: bool = False,
            use_post_norm: bool = False,
            ff_activation: nn.Module = nn.GELU(),
            ff_dropout: float = 0.1,
            use_gated: bool = False,
            use_moe: bool = False,
            num_experts: int = 1,
            num_shared_experts: int = 0,
            moe_top_k: int = 1,
            use_moe_att: bool = False,
            router_amp: bool = False,
            router_dtype: torch.dtype = torch.float32,
            *args,
            **kwargs,
    ):
        super(ClassicTransformerLayer, self).__init__(*args, **kwargs)

        self.attention = self_attention

        if use_gated:
            if use_moe:
                self.ff = GatedMoeFeedForward(embed_dim, ff_dim, num_experts, ff_activation, top_k=moe_top_k,
                                              dropout=ff_dropout, num_shared_experts=num_shared_experts,
                                              router_amp=router_amp, router_dtype=router_dtype)
            else:
                self.ff = GatedFeedForward(embed_dim, ff_dim, ff_activation, dropout=ff_dropout)
        else:
            if use_moe:
                self.ff = MoeFeedForward(embed_dim, ff_dim, num_experts, ff_activation, top_k=moe_top_k,
                                         dropout=ff_dropout, num_shared_experts=num_shared_experts,
                                         router_amp=router_amp, router_dtype=router_dtype)
            else:
                self.ff = FeedForward(embed_dim, ff_dim, ff_activation, dropout=ff_dropout)

        if use_rms_norm:
            self.norm1 = nn.RMSNorm(embed_dim)
            self.norm2 = nn.RMSNorm(embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        self.use_post_norm = use_post_norm
        self.use_moe = use_moe
        self.moe_top_k = moe_top_k
        self.use_moe_att = use_moe_att

    def moe_router_loss(self):
        ff_router_loss = self.ff.router_loss() if self.use_moe else None
        att_router_loss = self.attention.router_loss() if self.use_moe_att and self.attention.router_loss is not None else None

        if ff_router_loss is not None and att_router_loss is not None:
            return (ff_router_loss + att_router_loss) / 2
        elif ff_router_loss is not None:
            return ff_router_loss
        elif att_router_loss is not None:
            return att_router_loss
        else:
            return None

    def active_parameters(self) -> list[nn.Parameter]:
        if not self.use_moe:
            return list(self.parameters())
        else:
            attn_params = list(self.attention.parameters()) + list(self.norm1.parameters())
            ff_norm_params = list(self.norm2.parameters())
            router_params = list(self.ff.router.parameters())
            active_expert_params = []
            for i in range(self.moe_top_k):
                active_expert_params.extend(list(self.ff.experts[i].parameters()))
            return attn_params + ff_norm_params + router_params + active_expert_params

    def update_max_len(self, max_seq_len: int):
        if self.attention.rope is not None:
            self.attention.rope.update_max_len(max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None) -> torch.Tensor:
        # First step, self-attention
        residual = x
        if not self.use_post_norm:
            x = self.norm1(x)
        x = self.attention(x, x, x, mask=mask, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)
        x = residual + x
        if self.use_post_norm:
            x = self.norm1(x)
        # Second step, Feed Forward network
        residual = x
        if not self.use_post_norm:
            x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        if self.use_post_norm:
            x = self.norm2(x)
        return x