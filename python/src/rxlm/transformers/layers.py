import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ff import FeedForward, GatedFeedForward
from .moe import MoeFeedForward, GatedMoeFeedForward


class ReactiveTransformerLayer(nn.Module):
    """Reactive Transformer layer - extending the classic Transformer layer with Memory Cross-Attention"""

    def __init__(
            self,
            embed_dim: int,
            ff_dim: int,
            self_attention: MultiHeadAttention,
            memory_cross_attention: MultiHeadAttention = None,
            use_rms_norm: bool = False,
            use_post_norm: bool = False,
            ff_activation: nn.Module = nn.GELU(),
            ff_dropout: float = 0.1,
            use_gated: bool = False,
            use_moe: bool = False,
            num_experts: int = 1,
            moe_top_k: int = 1,
            use_moe_att: bool = False,
            skip_memory_cross_attention: bool = False,
            *args,
            **kwargs,
    ):
        super(ReactiveTransformerLayer, self).__init__(*args, **kwargs)

        self.attention = self_attention

        self.skip_memory_cross_attention = skip_memory_cross_attention
        self.memory_cross_attention = memory_cross_attention

        assert (not self.skip_memory_cross_attention and self.memory_cross_attention is not None) or self.skip_memory_cross_attention, \
            'Memory Cross Attention requires skip_memory_cross_attention=True'

        if use_gated:
            if use_moe:
                self.ff = GatedMoeFeedForward(embed_dim, ff_dim, num_experts, ff_activation, top_k=moe_top_k,
                                              dropout=ff_dropout)
            else:
                self.ff = GatedFeedForward(embed_dim, ff_dim, ff_activation, dropout=ff_dropout)
        else:
            if use_moe:
                self.ff = MoeFeedForward(embed_dim, ff_dim, num_experts, ff_activation, top_k=moe_top_k,
                                         dropout=ff_dropout)
            else:
                self.ff = FeedForward(embed_dim, ff_dim, ff_activation, dropout=ff_dropout)

        if use_rms_norm:
            self.norm1 = nn.RMSNorm(embed_dim)
            self.norm3 = nn.RMSNorm(embed_dim)

            if not self.skip_memory_cross_attention:
                self.norm2 = nn.RMSNorm(embed_dim)
                self.stm_norm = nn.RMSNorm(embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm3 = nn.LayerNorm(embed_dim)

            if not self.skip_memory_cross_attention:
                self.norm2 = nn.LayerNorm(embed_dim)
                self.stm_norm = nn.LayerNorm(embed_dim)

        self.use_post_norm = use_post_norm
        self.use_moe = use_moe
        self.moe_top_k = moe_top_k
        self.use_moe_att = use_moe_att

    def trainable_cross_attention_(self, is_trainable: bool, with_norms: bool = True):
        if not self.skip_memory_cross_attention:
            for param in self.memory_cross_attention.parameters():
                param.requires_grad_(is_trainable)
            if with_norms:
                for param in self.norm2.parameters():
                    param.requires_grad_(is_trainable)
                for param in self.stm_norm.parameters():
                    param.requires_grad_(is_trainable)

    def memory_parameters(self) -> list[nn.Parameter]:
        if not self.skip_memory_cross_attention:
            return list(self.memory_cross_attention.parameters()) + list(self.norm2.parameters()) + list(self.stm_norm.parameters())
        else:
            return []

    def not_memory_parameters(self) -> list[nn.Parameter]:
        return (list(self.attention.parameters()) + list(self.norm1.parameters()) +
                list(self.norm3.parameters()) + list(self.ff.parameters()))

    def active_parameters(self) -> list[nn.Parameter]:
        if not self.use_moe:
            return list(self.parameters())
        else:
            mem_params = self.memory_parameters()
            attn_params = list(self.attention.parameters()) + list(self.norm1.parameters())
            ff_norm_params = list(self.norm3.parameters())
            router_params = list(self.ff.router.parameters())
            active_expert_params = []
            for i in range(self.moe_top_k):
                active_expert_params.extend(list(self.ff.experts[i].parameters()))
            return mem_params + attn_params + ff_norm_params + router_params + active_expert_params

    def update_max_len(self, max_seq_len: int):
        if self.attention.rope is not None:
            self.attention.rope.update_max_len(max_seq_len)
        # if self.memory_cross_attention.rope is not None:
        #     self.memory_cross_attention.rope.update_max_len(max_seq_len)

    def moe_router_loss(self):
        ff_router_loss = self.ff.router_loss() if self.use_moe else None
        att_router_loss = None
        if self.use_moe_att:
            if self.attention.router_loss is not None and self.memory_cross_attention.router_loss is not None:
                att_router_loss = (self.attention.router_loss() + self.memory_cross_attention.router_loss()) / 2
            elif self.attention.router_loss is not None:
                att_router_loss = self.attention.router_loss()
            elif self.memory_cross_attention is not None and self.memory_cross_attention.router_loss is not None:
                att_router_loss = self.memory_cross_attention.router_loss()

        if ff_router_loss is not None and att_router_loss is not None:
            return (ff_router_loss + att_router_loss) / 2
        elif ff_router_loss is not None:
            return ff_router_loss
        elif att_router_loss is not None:
            return att_router_loss
        else:
            return None

    def forward(self, x: torch.Tensor, stm: torch.Tensor, mask: torch.Tensor = None, stm_kv_cache: tuple[torch.Tensor, torch.Tensor] = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None) -> torch.Tensor:
        # First step, self-attention
        residual = x
        if not self.use_post_norm:
            x = self.norm1(x)
        x = self.attention(x, x, x, mask=mask, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)

        x = residual + x
        if self.use_post_norm:
            x = self.norm1(x)

        # Second step, Memory cross-attention
        if not self.skip_memory_cross_attention:
            residual = x
            if not self.use_post_norm:
                x = self.norm2(x)

            # normalize STM and prepare STM mask
            stm = self.stm_norm(stm)
            mem_mask = mask.squeeze(1).unsqueeze(-1).expand(-1, -1, -1, stm.size(1)) \
                if mask is not None else None

            x = self.memory_cross_attention(x, stm, stm, mask=mem_mask, stm_kv_cache=stm_kv_cache, current_positions=current_positions)
            x = residual + x
            if self.use_post_norm:
                x = self.norm2(x)

        # Third step, Feed Forward network
        residual = x
        if not self.use_post_norm:
            x = self.norm3(x)
        x = self.ff(x)
        x = residual + x
        if self.use_post_norm:
            x = self.norm3(x)
        return x
