import torch
import torch.nn as nn
from .positional import AbsolutePositionalEmbedding
from .mask import create_causal_mask
from ..memory.stm import ShortTermMemory


class ReactiveTransformerBase(nn.Module):
    """Base class for Reactive Transformer models - common logic for both decoders and encoders."""

    def __init__(
            self,
            stm: ShortTermMemory,
            embedding: nn.Embedding,
            own_layers: nn.ModuleList,
            shared_layers: nn.ModuleList = None,
            absolute_embedding: AbsolutePositionalEmbedding = None,
            use_flash_attention: bool = False,
            use_relative_embedding: bool = False,
            use_moe: bool = False,
            *args,
            **kwargs,
    ):
        super(ReactiveTransformerBase, self).__init__(*args, **kwargs)

        self.embedding = embedding
        self.stm = stm
        self.pos_embedding = absolute_embedding
        self.use_flash_attention = use_flash_attention
        self.use_relative_embedding = use_relative_embedding

        self.shared_layers = shared_layers
        self.layers = own_layers
        self.num_shared_layers = len(shared_layers) if shared_layers else 0
        self.num_own_layers = len(own_layers) if own_layers else 0
        self.use_moe = use_moe

    def trainable_cross_attention_(self, is_trainable: bool, with_norms: bool = True):
        for i in range(self.num_shared_layers):
            self.shared_layers[i].trainable_cross_attention_(is_trainable, with_norms)
        for i in range(self.num_own_layers):
            self.layers[i].trainable_cross_attention_(is_trainable, with_norms)

    def memory_parameters(self) -> list[nn.Parameter]:
        own = [param for layer in self.layers for param in layer.memory_parameters()]
        shared = [param for layer in self.shared_layers for param in layer.memory_parameters()] if self.shared_layers else []
        return own + shared

    def not_memory_parameters(self) -> list[nn.Parameter]:
        own = [param for layer in self.layers for param in layer.not_memory_parameters()]
        shared = [param for layer in self.shared_layers for param in layer.not_memory_parameters()] if self.shared_layers else []
        return own + shared

    def moe_router_loss(self):
        if self.use_moe:
            return torch.stack([self.layers[i].moe_router_loss() for i in range(self.num_own_layers) if self.layers[i].use_moe or self.layers[i].use_moe_att] + [
                self.shared_layers[i].moe_router_loss() for i in range(self.num_shared_layers) if self.shared_layers[i].use_moe or self.shared_layers[i].use_moe_att]).mean()
        else:
            return None

    def _handle_layer(self, i: int, x: torch.Tensor, mask: torch.Tensor = None, is_shared: bool = False, stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None):
        stm_layer_idx = i if is_shared else i + self.num_shared_layers
        layer_stm = self.stm(stm_layer_idx)
        # expand layer STM to batch size, if it's not in batch mode
        if layer_stm.size(0) == 1:
            layer_stm = layer_stm.expand(x.size(0), -1, -1)
        layer = self.shared_layers[i] if is_shared else self.layers[i]
        layer_stm_cache = stm_kv_cache[stm_layer_idx] if stm_kv_cache is not None else None
        return layer(x, layer_stm, mask=mask, stm_kv_cache=layer_stm_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared logic for encoders and decoders - apply embeddings and positional encoding
        x = self.embedding(x)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)
        return x


class ReactiveTransformerDecoder(ReactiveTransformerBase):
    """Reactive Transformer decoder - extending the classic Transformer decoder with Memory Cross-Attention"""

    def __init__(self, embed_dim: int, vocab_size: int, use_head_norm: bool = False, init_identity_norm: bool = False, *args, **kwargs):
        super(ReactiveTransformerDecoder, self).__init__(*args, **kwargs)

        self.head = nn.Linear(embed_dim, vocab_size)
        self.use_head_norm = use_head_norm
        if use_head_norm:
            self.head_norm = nn.LayerNorm(embed_dim)
            if init_identity_norm:
                self.head_norm.weight.data.fill_(1.0)
                self.head_norm.bias.data.fill_(0.0)
        else:
            self.head_norm = None

    def not_memory_parameters(self) -> list[nn.Parameter]:
        layer_params = super().not_memory_parameters()
        head_params = list(self.head.parameters())
        if self.use_head_norm:
            head_params += list(self.head_norm.parameters())
        return layer_params + head_params

    def prepare_stm_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        stm_kv_cache = []

        for i in range(self.num_shared_layers):
            layer_stm = self.stm(i)
            normalized_layer_stm = self.shared_layers[i].stm_norm(layer_stm)
            projected_key = self.shared_layers[i].memory_cross_attention.k_proj(normalized_layer_stm)
            projected_value = self.shared_layers[i].memory_cross_attention.v_proj(normalized_layer_stm)

            b, t, d = layer_stm.size()
            mapped_key = self.shared_layers[i].memory_cross_attention.split_kv_head(projected_key, b, t, d)
            mapped_value = self.shared_layers[i].memory_cross_attention.split_kv_head(projected_value, b, t, d)

            stm_kv_cache.append((mapped_key, mapped_value))

        for i in range(self.num_own_layers):
            layer_stm = self.stm(i + self.num_shared_layers)
            normalized_layer_stm = self.layers[i].stm_norm(layer_stm)
            projected_key = self.layers[i].memory_cross_attention.k_proj(normalized_layer_stm)
            projected_value = self.layers[i].memory_cross_attention.v_proj(normalized_layer_stm)

            b, t, d = layer_stm.size()
            mapped_key = self.layers[i].memory_cross_attention.split_kv_head(projected_key, b, t, d)
            mapped_value = self.layers[i].memory_cross_attention.split_kv_head(projected_value, b, t, d)

            stm_kv_cache.append((mapped_key, mapped_value))

        return stm_kv_cache

    def reset_self_attn_cache(self):
        for i in range(self.num_shared_layers):
            self.shared_layers[i].attention.reset_inner_cache()
        for i in range(self.num_own_layers):
            self.layers[i].attention.reset_inner_cache()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None) -> torch.Tensor:
        x = super().forward(x)  # apply embeddings
        seq_len = x.size(1)
        if not self.use_flash_attention and self.use_relative_embedding:
            mask = create_causal_mask(seq_len, device=x.device)
            if attention_mask is not None:
                mask &= attention_mask.unsqueeze(1).unsqueeze(1).bool()
        elif attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        else:
            mask = None
        # Process shared layers
        if self.shared_layers is not None:
            for i in range(self.num_shared_layers):
                x = self._handle_layer(i, x, mask=mask, is_shared=True, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)
        # Process own layers
        for i in range(self.num_own_layers):
            x = self._handle_layer(i, x, mask=mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)
        return self.head(self.head_norm(x) if self.use_head_norm else x)


class ReactiveTransformerEncoder(ReactiveTransformerBase):
    """Reactive Transformer encoder - extending the classic Transformer encoder with Memory Cross-Attention"""

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)  # apply embeddings
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        hidden_states = []
        # Process shared layers
        if self.shared_layers is not None:
            for i in range(self.num_shared_layers):
                x = self._handle_layer(i, x, mask=attention_mask, is_shared=True)
                hidden_states.append(x)
        # Process own layers
        for i in range(self.num_own_layers):
            x = self._handle_layer(i, x, mask=attention_mask)
            hidden_states.append(x)
        return x, torch.stack(hidden_states)


