import torch
from torch import nn
from typing import TypedDict, Union
from huggingface_hub import PyTorchModelHubMixin
from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.attention import init_attention
from ..transformers.layers import ClassicTransformerLayer
from ..transformers.models import ClassicTransformerDecoder
from ..transformers.ff import get_activation_layer
from ..utils import get_model_size
from .attention import init_experimental_attention


class ExperimentalAttentionTransformerConfig(TypedDict):
    num_layers: int
    vocab_size: int
    embed_dim: int
    ff_dim: int
    att_heads: int
    seq_len: int
    use_flash_attention: bool
    use_gated: bool
    ff_activation: str
    ff_dropout: float
    att_dropout: float
    use_rms_norm: bool
    att_groups: int
    use_moe_ff: bool
    ff_num_experts: int
    ff_moe_top_k: int
    att_type: str
    att_num_experts: int
    att_num_query_experts: int
    att_num_query_groups: int
    att_num_global_tokens: int
    att_window_size: int
    use_head_norm: bool
    init_identity_norm: bool


class ExperimentalAttentionTransformer(nn.Module, PyTorchModelHubMixin, pipeline_tag="text-generation", license="apache-2.0"):
    """
    Research model for experiments with new attention layers.

    Currently, accepts SparseQueryAttention, GroupedMoeAttention, DeepMoeAttention and standard variants (MHA/GQA/MQA) for reference models
    """

    def __init__(
            self,
            num_layers: int = 6,
            vocab_size: int = 5000,
            embed_dim: int = 128,
            ff_dim: int = 384,
            att_heads: int = 16,
            seq_len: int = 256,
            use_flash_attention: bool = True,
            use_gated: bool = True,
            ff_activation: str = "swish",
            ff_dropout: float = 0.0,
            att_dropout: float = 0.0,
            use_rms_norm: bool = True,
            att_groups: int = 1,
            use_moe_ff: bool = False,
            ff_num_experts: int = 1,
            ff_moe_top_k: int = 1,
            att_type: str = 'gma',
            att_num_experts: int = None,
            att_num_query_experts: int = None,
            att_num_query_groups: int = None,
            att_num_global_tokens: int = 16,
            att_window_size: int = 128,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
            **kwargs
    ):
        super(ExperimentalAttentionTransformer, self).__init__(**kwargs)
        assert ff_activation in ['relu', 'gelu',
                                 'swish', 'silu', 'linear',
                                 'sigmoid'], 'Feed-forward activation could be "relu", "gelu", "swish", "silu", "linear", "sigmoid".'
        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma', 'sqa', 'flex', 'flex-sqa'], 'Self-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa", "flex", "flex-sqa".'

        embedding = nn.Embedding(vocab_size, embed_dim)
        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)

        ff_activation = get_activation_layer(ff_activation)

        if att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            att_init = lambda: init_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                              use_flash_attention=use_flash_attention, dropout=att_dropout,
                                              max_seq_len=seq_len, is_causal=True, num_query_groups=att_num_query_groups)
        else:
            att_init = lambda: init_experimental_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=True, num_experts=att_num_experts,
                                                           num_query_experts=att_num_query_experts, num_query_groups=att_num_query_groups,
                                                           num_global_tokens=att_num_global_tokens, window_size=att_window_size)

        use_moe_att = att_type in ['gma', 'dma']

        self.model = ClassicTransformerDecoder(
            embed_dim,
            vocab_size,
            embedding=embedding,
            layers=nn.ModuleList([
                ClassicTransformerLayer(
                    embed_dim,
                    ff_dim,
                    use_gated=use_gated,
                    use_moe=use_moe_ff,
                    num_experts=ff_num_experts,
                    moe_top_k=ff_moe_top_k,
                    ff_activation=ff_activation,
                    ff_dropout=ff_dropout,
                    use_rms_norm=use_rms_norm,
                    self_attention=att_init(),
                    use_moe_att=use_moe_att,
                    use_head_norm=use_head_norm,
                    init_identity_norm=init_identity_norm,
                ) for _ in range(num_layers)
            ]),
            use_flash_attention=use_flash_attention,
        )

    def params_count(self):
        return get_model_size(self.model)

    def load_shared_embedding(self, embedding: nn.Embedding):
        self.model.embedding = embedding

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)
