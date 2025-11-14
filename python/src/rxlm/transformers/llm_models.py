import torch
import torch.nn as nn
from .positional import AbsolutePositionalEmbedding
from .mask import create_causal_mask


class ClassicTransformerBase(nn.Module):
    """Base class for Classic Transformer models - common logic for both decoders and encoders."""

    def __init__(
            self,
            embedding: nn.Embedding,
            layers: nn.ModuleList,
            absolute_embedding: AbsolutePositionalEmbedding = None,
            use_flash_attention: bool = False,
            use_relative_embedding: bool = False,
            *args,
            **kwargs,
    ):
        super(ClassicTransformerBase, self).__init__(*args, **kwargs)

        self.embedding = embedding
        self.pos_embedding = absolute_embedding
        self.use_flash_attention = use_flash_attention
        self.use_relative_embedding = use_relative_embedding

        self.layers = layers
        self.num_layers = len(layers) if layers else 0

    def update_max_len(self, max_seq_len: int):
        self.layers[0].update_max_len(max_seq_len)

    def moe_router_loss(self):
        return torch.stack([self.layers[i].moe_router_loss() for i in range(self.num_layers) if self.layers[i].use_moe or self.layers[i].use_moe_att]).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared logic for encoders and decoders - apply embeddings and positional encoding
        x = self.embedding(x)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)
        return x


class ClassicTransformerDecoder(ClassicTransformerBase):
    """Classic Transformer decoder - for decoder-only Transformer models"""

    def __init__(self, embed_dim: int, vocab_size: int, use_head_norm: bool = False, init_identity_norm: bool = False, tie_embeddings: bool = False, head_norm_type: str = 'layer_norm', *args, **kwargs):
        super(ClassicTransformerDecoder, self).__init__(*args, **kwargs)

        self.tie_embeddings = tie_embeddings

        # When using tied embeddings, we don't create a head at all
        # In forward pass, we'll compute logits as x @ embedding.weight.T (no bias)
        if not tie_embeddings:
            self.head = nn.Linear(embed_dim, vocab_size)
        else:
            self.head = None

        self.use_head_norm = use_head_norm
        if use_head_norm:
            if head_norm_type == 'rms_norm':
                self.head_norm = nn.RMSNorm(embed_dim)
            elif head_norm_type == 'layer_norm':
                self.head_norm = nn.LayerNorm(embed_dim)
                if init_identity_norm:
                    self.head_norm.weight.data.fill_(1.0)
                    self.head_norm.bias.data.fill_(0.0)
            else:
                raise ValueError(f"head_norm_type must be 'layer_norm' or 'rms_norm', got '{head_norm_type}'")
        else:
            self.head_norm = None

    def reset_self_attn_cache(self):
        for i in range(self.num_layers):
            self.layers[i].attention.reset_inner_cache()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None) -> torch.Tensor:
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

        # Process layers
        for i in range(self.num_layers):
            x = self.layers[i](x, mask=mask, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)

        # Apply head normalization if enabled
        if self.use_head_norm:
            x = self.head_norm(x)

        # Compute logits with tied embeddings or regular head
        if self.tie_embeddings:
            # Manual matmul with transposed embedding weights (no bias for symmetry)
            return torch.matmul(x, self.embedding.weight.T)
        else:
            return self.head(x)


class ClassicTransformerEncoder(ClassicTransformerBase):
    """Classic Transformer encoder - for encoder-only Transformer models"""

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)  # apply embeddings
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        hidden_states = []
        # Process own layers
        for i in range(self.num_own_layers):
            x = self.layers[i](x, mask=attention_mask)
            hidden_states.append(x)
        return x, torch.stack(hidden_states)
