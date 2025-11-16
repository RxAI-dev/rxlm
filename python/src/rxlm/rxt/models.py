import torch
from torch import nn
from typing import TypedDict, Union, Optional, Iterator, Literal
from enum import Enum
from huggingface_hub import PyTorchModelHubMixin
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.attention import init_attention
from ..transformers.layers import ReactiveTransformerLayer
from ..transformers.models import ReactiveTransformerBase, ReactiveTransformerEncoder, ReactiveTransformerDecoder
from ..transformers.ff import get_activation_layer
from ..transformers.sampler import sample, sample_batch
from ..transformers.llm_layers import ClassicTransformerLayer
from ..memory.stm import ShortTermMemory
from ..memory.norm import init_memory_norm
from ..memory.attention import StmMemoryAttention, InterlayerStmMemoryAttention, SelfStmMemoryAttention, SelfInterlayerStmMemoryAttention
from ..memory.gate import ResidualGate, ResidualGateType, SlotStatusType
from ..utils import get_model_size
from ..experimental.attention import init_experimental_attention
from ..training.tokenizer import decode_post_process


class RxTComponentConfig(TypedDict):
    num_layers: int
    vocab_size: int
    embed_dim: int
    ff_dim: int
    att_heads: int
    seq_len: int
    stm_size: int
    use_flash_attention: bool
    use_gated: bool
    ff_activation: str
    ff_dropout: float
    att_dropout: float
    use_rms_norm: bool
    att_groups: int
    use_moe: bool
    num_experts: int
    moe_top_k: int
    self_att_type: str
    cross_att_type: str
    att_experts: int
    att_query_experts: int
    att_query_groups: int
    cross_att_groups: int
    cross_att_query_groups: int
    use_head_norm: bool
    init_identity_norm: bool


class RxTComponentBase(nn.Module):
    """Base class for RxT-Alpha (Reactive Transformer) components (encoder and decoder)"""

    def __init__(
            self,
            is_causal: bool,
            num_layers: int = 12,
            vocab_size: int = 20000,
            embed_dim: int = 512,
            ff_dim: int = 1536,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            use_gated: bool = True,
            ff_activation: str = "swish",
            ff_dropout: float = 0.0,
            att_dropout: float = 0.0,
            use_rms_norm: bool = True,
            att_groups: int = 1,
            use_moe: bool = False,
            num_experts: int = 1,
            moe_top_k: int = 1,
            self_att_type: str = 'gqa',
            cross_att_type: str = 'mqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            cross_att_groups: int = None,
            cross_att_query_groups: int = None,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
            skip_memory_cross_attention: bool = False,
            stateless_layers_config: list[Literal['dense', 'moe']] = None,
            dense_layer_dim: int = 1536,
            **kwargs
    ):
        super(RxTComponentBase, self).__init__(**kwargs)
        assert ff_activation in ['relu', 'gelu',
                                 'swish', 'silu', 'linear',
                                 'sigmoid'], 'Feed-forward activation could be "relu", "gelu", "swish", "silu", "linear", "sigmoid".'
        assert self_att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                                 'sqa'], 'Self-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'
        assert cross_att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                                  'sqa'], 'Memory cross-attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        embedding = nn.Embedding(vocab_size, embed_dim)
        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        ff_activation = get_activation_layer(ff_activation)

        if self_att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            att_init = lambda: init_attention(embed_dim, att_heads, self_att_type, att_groups, rope=rope,
                                              use_flash_attention=use_flash_attention, dropout=att_dropout,
                                              max_seq_len=seq_len, is_causal=is_causal, num_query_groups=att_query_groups,
                                              )
        else:
            att_init = lambda: init_experimental_attention(embed_dim, att_heads, self_att_type, att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=is_causal,
                                                           num_experts=att_experts,
                                                           num_query_experts=att_query_experts,
                                                           num_query_groups=att_query_groups)


        if not skip_memory_cross_attention:
            if cross_att_type in ['mha', 'gqa', 'mqa', 'sqa']:
                cross_att_init = lambda: init_attention(embed_dim, att_heads, cross_att_type, att_groups, rope=rope,
                                                        use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                        max_seq_len=seq_len, is_causal=False, rope_only_for_query=True,
                                                        num_query_groups=cross_att_query_groups or att_query_groups)
            else:
                cross_att_init = lambda: init_experimental_attention(embed_dim, att_heads, cross_att_type,
                                                                     cross_att_groups or att_groups, rope=rope,
                                                                     use_flash_attention=use_flash_attention,
                                                                     dropout=att_dropout,
                                                                     max_seq_len=seq_len, is_causal=False,
                                                                     num_experts=att_experts,
                                                                     num_query_experts=att_query_experts,
                                                                     num_query_groups=cross_att_query_groups or att_query_groups,
                                                                     rope_only_for_query=True)

            if stateless_layers_config is not None:
                def stateless_layer_init(layer_type: Literal['dense', 'moe']):
                    if layer_type == 'dense':
                        return ClassicTransformerLayer(
                            embed_dim,
                            dense_layer_dim,
                            use_gated=use_gated,
                            use_moe=False,
                            ff_activation=ff_activation,
                            ff_dropout=ff_dropout,
                            use_rms_norm=use_rms_norm,
                            self_attention=att_init(),
                        )
                    else:
                        return ClassicTransformerLayer(
                            embed_dim,
                            ff_dim,
                            use_gated=use_gated,
                            use_moe=use_moe,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            ff_activation=ff_activation,
                            ff_dropout=ff_dropout,
                            use_rms_norm=use_rms_norm,
                            self_attention=att_init(),
                        )

                stateless_layers = nn.ModuleList([
                    stateless_layer_init(layer_type) for layer_type in stateless_layers_config
                ])
            else:
                stateless_layers = None

            layers = nn.ModuleList([
                ReactiveTransformerLayer(
                    embed_dim,
                    ff_dim,
                    use_gated=use_gated,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    ff_activation=ff_activation,
                    ff_dropout=ff_dropout,
                    use_rms_norm=use_rms_norm,
                    self_attention=att_init(),
                    memory_cross_attention=cross_att_init(),
                ) for _ in range(num_layers)
            ])
        else:
            layers = nn.ModuleList([
                ReactiveTransformerLayer(
                    embed_dim,
                    ff_dim,
                    use_gated=use_gated,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    ff_activation=ff_activation,
                    ff_dropout=ff_dropout,
                    use_rms_norm=use_rms_norm,
                    self_attention=att_init(),
                    memory_cross_attention=None,
                    skip_memory_cross_attention=skip_memory_cross_attention,
                ) for _ in range(num_layers)
            ])
            stateless_layers = None

        self.model = self._init_model(
            stm, layers, embedding, use_flash_attention, embed_dim, vocab_size, use_moe,
            use_head_norm=use_head_norm, init_identity_norm=init_identity_norm, stateless_layers=stateless_layers
        )

    def _init_model(self, stm: ShortTermMemory, layers: nn.ModuleList, embedding: nn.Embedding,
                    use_flash_attention: bool, embed_dim: int, vocab_size: int, use_moe: bool,
                    use_head_norm: bool = False, init_identity_norm: bool = False, stateless_layers: nn.ModuleList = None) -> ReactiveTransformerBase:
        pass

    def params_count(self):
        return get_model_size(self.model)

    def load_shared_embedding(self, embedding: nn.Embedding):
        self.model.embedding = embedding

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def memory_parameters(self) -> list[nn.Parameter]:
        return self.model.memory_parameters()

    def not_memory_parameters(self) -> list[nn.Parameter]:
        return self.model.not_memory_parameters()

    def active_parameters(self) -> list[nn.Parameter]:
        return self.model.active_parameters()

    def freeze_without_memory(self, unfreeze_norms: bool = True):
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.trainable_cross_attention_(True, with_norms=unfreeze_norms)

    def freeze_memory(self, with_norms: bool = True):
        self.model.trainable_cross_attention_(False, with_norms=with_norms)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

    def unfreeze_all(self, freeze_memory: bool = False, freeze_memory_norms: bool = True):
        for param in self.model.parameters():
            param.requires_grad_(True)

        if freeze_memory:
            self.freeze_memory(with_norms=freeze_memory_norms)

    def update_max_len(self, max_seq_len: int):
        for layer in self.model.layers:
            layer.update_max_len(max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.model(x, attention_mask=attention_mask)


class RxTEncoderComponent(RxTComponentBase):
    """RxT-Alpha (Reactive Transformer) encoder model"""

    def __init__(self, **kwargs: RxTComponentConfig):
        super(RxTEncoderComponent, self).__init__(False, **kwargs)

    def _init_model(
            self,
            stm: ShortTermMemory,
            layers: nn.ModuleList,
            embedding: nn.Embedding,
            use_flash_attention: bool,
            embed_dim: int,
            vocab_size: int,
            use_moe: bool,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
            stateless_layers: nn.ModuleList = None,
    ) -> ReactiveTransformerEncoder:
        return ReactiveTransformerEncoder(
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
            use_moe=use_moe,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, attention_mask=attention_mask)


class RxTDecoderComponent(RxTComponentBase):
    """RxT-Alpha (Reactive Transformer) decoder model"""

    def __init__(self, **kwargs):
        super(RxTDecoderComponent, self).__init__(True, **kwargs)

    def _init_model(
            self, stm: ShortTermMemory,
            layers: nn.ModuleList,
            embedding: nn.Embedding,
            use_flash_attention: bool,
            embed_dim: int,
            vocab_size: int,
            use_moe: bool,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
            stateless_layers: nn.ModuleList = None,
    ) -> ReactiveTransformerDecoder:
        return ReactiveTransformerDecoder(
            embed_dim,
            vocab_size,
            stm=stm,
            embedding=embedding,
            own_layers=layers,
            use_flash_attention=use_flash_attention,
            use_moe=use_moe,
            use_head_norm=use_head_norm,
            init_identity_norm=init_identity_norm,
            stateless_layers=stateless_layers,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, attention_mask=attention_mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)


class RxTSimpleMemoryAttentionComponent(nn.Module):
    """RxT-Alpha (Reactive Transformer) memory attention model"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            disable_residual: bool = False,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTSimpleMemoryAttentionComponent, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            att_init = lambda: init_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                              use_flash_attention=use_flash_attention, dropout=att_dropout,
                                              max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True,
                                              num_query_groups=att_query_groups)
        else:
            att_init = lambda: init_experimental_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=False,
                                                           num_experts=att_experts,
                                                           num_query_experts=att_query_experts,
                                                           num_query_groups=att_query_groups, rope_only_for_keys=True)

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
                disable_residual=disable_residual,
            ) for _ in range(num_layers)
        ])

        self.model = StmMemoryAttention(
            stm, attention_layers, memory_norm_layers,
            memory_input_norm_layers, residual_gates,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)


class RxTInterlayerMemoryAttentionComponent(nn.Module):
    """RxT-Alpha (Reactive Transformer) memory attention model with interlayer STM attention"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            interlayer_att_dropout: float = 0.0,
            interlayer_att_groups: int = 1,
            interlayer_att_type: str = 'sqa',
            interlayer_att_experts: int = None,
            interlayer_att_query_experts: int = None,
            interlayer_att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTInterlayerMemoryAttentionComponent, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            att_init = lambda: init_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True,
                num_query_groups=att_query_groups
            )
        else:
            att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, num_experts=att_experts,
                num_query_experts=att_query_experts, num_query_groups=att_query_groups,
                rope_only_for_keys=True
            )

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        # Interlayer attention
        if interlayer_att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            interlayer_att_init = lambda: init_attention(
                embed_dim, att_heads, interlayer_att_type, interlayer_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=interlayer_att_dropout, is_causal=False,
                num_query_groups=interlayer_att_query_groups
            )
        else:
            interlayer_att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, interlayer_att_type, interlayer_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=interlayer_att_dropout, is_causal=False,
                num_experts=interlayer_att_experts, num_query_experts=interlayer_att_query_experts, num_query_groups=interlayer_att_query_groups
            )

        mean_attention_layers = nn.ModuleList([interlayer_att_init() for _ in range(num_layers)])

        mean_stm_norm = init_memory_norm(
            norm_type, embed_dim, stm_size, decay=norm_decay,
            init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale
        )

        mean_memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])

        mean_residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        self.model = InterlayerStmMemoryAttention(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gates,
            mean_attention_layers, mean_memory_norm_layers, mean_residual_gates, mean_stm_norm,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)

class RxTSelfMemoryAttentionComponent(nn.Module):
    """RxT-Alpha (Reactive Transformer) memory attention model with STM layer self-attention"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            self_att_dropout: float = 0.0,
            self_att_groups: int = 1,
            self_att_type: str = 'sqa',
            self_att_experts: int = None,
            self_att_query_experts: int = None,
            self_att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            use_gate_for_self_attention: bool = False,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTSelfMemoryAttentionComponent, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            att_init = lambda: init_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True,
                num_query_groups=att_query_groups,
            )
        else:
            att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, num_experts=att_experts,
                num_query_experts=att_query_experts, num_query_groups=att_query_groups,
                rope_only_for_keys=True
            )

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        # Self attention
        if self_att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            self_att_init = lambda: init_attention(
                embed_dim, att_heads, self_att_type, self_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=self_att_dropout,
                is_causal=False, num_query_groups=self_att_query_groups
            )
        else:
            self_att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, self_att_type, self_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=self_att_dropout, is_causal=False,
                num_experts=self_att_experts, num_query_experts=self_att_query_experts, num_query_groups=self_att_query_groups
            )

        self_attention_layers = nn.ModuleList([self_att_init() for _ in range(num_layers)])

        self_memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])

        self_residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gate_for_self_attention, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        self.model = SelfStmMemoryAttention(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gates,
            self_attention_layers, self_memory_norm_layers, self_residual_gates,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)


class RxTSelfInterlayerMemoryAttentionComponent(nn.Module):
    """RxT-Alpha (Reactive Transformer) memory attention model with interlayer STM attention"""

    def __init__(
            self,
            num_layers: int = 12,
            embed_dim: int = 512,
            att_heads: int = 16,
            seq_len: int = 1024,
            stm_size: int = 1024,
            use_flash_attention: bool = False,
            att_dropout: float = 0.0,
            att_groups: int = 1,
            att_type: str = 'sqa',
            att_experts: int = None,
            att_query_experts: int = None,
            att_query_groups: int = None,
            interlayer_att_dropout: float = 0.0,
            interlayer_att_groups: int = 1,
            interlayer_att_type: str = 'sqa',
            interlayer_att_experts: int = None,
            interlayer_att_query_experts: int = None,
            interlayer_att_query_groups: int = None,
            norm_type: str = 'classic-rms',
            norm_init_gate: float = -2.0,
            norm_per_dim_scale: bool = False,
            norm_decay: float = 0.9,
            use_gated_residual: bool = False,
            residual_per_slot_gate: bool = True,
            residual_gate_init: float = 3.0,
            residual_gate_type: ResidualGateType = 'static',
            residual_gate_slot_status_type: SlotStatusType = 'mean',
            use_tanh_residual_gate: bool = True,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs,
    ):
        super(RxTSelfInterlayerMemoryAttentionComponent, self).__init__(**kwargs)

        assert att_type in ['mha', 'gqa', 'mqa', 'gma', 'dma',
                            'sqa'], 'Memory attention type could be "mha", "gqa", "mqa", "gma", "dma", "sqa".'

        rope = RotaryPositionalEmbedding(embed_dim // att_heads, seq_len)
        stm = ShortTermMemory(num_layers, embed_dim, stm_size)

        if att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            att_init = lambda: init_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, rope_only_for_keys=True, num_query_groups=att_query_groups,
            )
        else:
            att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, att_type, att_groups, rope=rope,
                use_flash_attention=use_flash_attention, dropout=att_dropout,
                max_seq_len=seq_len, is_causal=False, num_experts=att_experts,
                num_query_experts=att_query_experts, num_query_groups=att_query_groups,
                rope_only_for_keys=True
            )

        memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])
        memory_input_norm_layers = nn.ModuleList(nn.RMSNorm(embed_dim) for _ in range(num_layers))
        attention_layers = nn.ModuleList([att_init() for _ in range(num_layers)])
        residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        # Interlayer attention
        if interlayer_att_type in ['mha', 'gqa', 'mqa', 'sqa']:
            interlayer_att_init = lambda: init_attention(
                embed_dim, att_heads, interlayer_att_type, interlayer_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=interlayer_att_dropout, is_causal=False, num_query_groups=interlayer_att_query_groups
            )
        else:
            interlayer_att_init = lambda: init_experimental_attention(
                embed_dim, att_heads, interlayer_att_type, interlayer_att_groups, rope=None,
                use_flash_attention=use_flash_attention, dropout=interlayer_att_dropout, is_causal=False,
                num_experts=interlayer_att_experts, num_query_experts=interlayer_att_query_experts, num_query_groups=interlayer_att_query_groups
            )

        mean_attention_layers = nn.ModuleList([interlayer_att_init() for _ in range(num_layers)])

        mean_stm_norm = init_memory_norm(
            norm_type, embed_dim, stm_size, decay=norm_decay,
            init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale
        )

        mean_memory_norm_layers = nn.ModuleList([init_memory_norm(norm_type, embed_dim, stm_size, decay=norm_decay,
                                                             init_gate=norm_init_gate, per_dim_scale=norm_per_dim_scale)
                                            for _ in range(num_layers)])

        mean_residual_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        interlayer_gates = nn.ModuleList([
            ResidualGate(
                stm_size, embed_dim,
                use_gate=use_gated_residual, gate_type=residual_gate_type,
                per_slot_gate=residual_per_slot_gate, init_gate=residual_gate_init,
                use_tanh_gate=use_tanh_residual_gate, slot_status_type=residual_gate_slot_status_type,
            ) for _ in range(num_layers)
        ])

        self.model = SelfInterlayerStmMemoryAttention(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gates,
            mean_attention_layers, mean_memory_norm_layers, mean_residual_gates, interlayer_gates, mean_stm_norm,
            debug_mode=debug_mode, debug_interval=debug_interval,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_shared_memory(self, stm: ShortTermMemory):
        self.model.stm = stm

    def update_max_len(self, max_seq_len: int):
        self.model.update_max_len(max_seq_len)

    def reset_memory(self, init_type: str = None):
        self.model.stm.reset(init_type)

    def clone_reset_memory(self):
        self.model.stm.clone_detach_reset()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)


# Serializable component models
class RxTDecoder(RxTDecoderComponent, PyTorchModelHubMixin, pipeline_tag="text-generation", license="apache-2.0"):
    pass


class RxTEncoder(RxTEncoderComponent, PyTorchModelHubMixin, pipeline_tag="fill-mask", license="apache-2.0"):
    pass


class RxTSimpleMemoryAttention(RxTSimpleMemoryAttentionComponent, PyTorchModelHubMixin, license="apache-2.0"):
    pass


class RxTInterlayerMemoryAttention(RxTInterlayerMemoryAttentionComponent, PyTorchModelHubMixin, license="apache-2.0"):
    pass


class RxTSelfMemoryAttention(RxTSelfMemoryAttentionComponent, PyTorchModelHubMixin, license="apache-2.0"):
    pass


class RxTSelfInterlayerMemoryAttention(RxTSelfInterlayerMemoryAttentionComponent, PyTorchModelHubMixin, license="apache-2.0"):
    pass


class RxTInterlayerMemoryAttentionConfig(TypedDict):
    num_layers: int
    embed_dim: int
    att_heads: int
    seq_len: int
    stm_size: int
    use_flash_attention: bool
    att_dropout: float
    att_groups: int
    att_type: str
    att_experts: int
    att_query_experts: int
    att_query_groups: int
    interlayer_att_dropout: float
    interlayer_att_groups: int
    interlayer_att_type: str
    interlayer_att_experts: int
    interlayer_att_query_experts: int
    interlayer_att_query_groups: int
    norm_type: str
    norm_init_gate: float
    norm_per_dim_scale: bool
    norm_decay: float
    use_gated_residual: bool
    residual_per_slot_gate: bool
    residual_gate_init: float
    residual_gate_type: ResidualGateType
    residual_gate_slot_status_type: SlotStatusType
    use_tanh_residual_gate: bool
    debug_mode: bool
    debug_interval: int

class RxTAlphaPretrainedConfig(TypedDict):
    decoder: RxTDecoder
    encoder: RxTEncoder
    memory_attention: RxTInterlayerMemoryAttention

class RxTAlphaTokenizerConfig(TypedDict):
    bos_token_id: int
    eos_token_id: int
    answer_token_id: int
    query_token_id: int
    pad_token_id: int

class RxTForwardAction(Enum):
    DECODE = 1
    UPDATE = 2

class RxTAlpha(nn.Module, PyTorchModelHubMixin, pipeline_tag="text-generation", license="apache-2.0"):
    def __init__(
            self,
            decoder_config: RxTComponentConfig,
            encoder_config: RxTComponentConfig,
            memory_attention_config: RxTInterlayerMemoryAttentionConfig,
            tokenizer_config: RxTAlphaTokenizerConfig,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            memory_attention_variant: Literal['interlayer', 'self-interlayer'] = 'interlayer',
            **kwargs,
    ):
        super(RxTAlpha, self).__init__(**kwargs)
        self.decoder_config = decoder_config
        self.encoder_config = encoder_config
        self.memory_attention_config = memory_attention_config

        self.decoder = RxTDecoderComponent(**decoder_config)
        self.encoder = RxTEncoderComponent(**encoder_config)

        if memory_attention_variant == 'interlayer':
            self.memory_attention = RxTInterlayerMemoryAttentionComponent(**memory_attention_config)
        else:
            self.memory_attention = RxTSelfInterlayerMemoryAttentionComponent(**memory_attention_config)

        self.batch_size = 1
        self.bos_token_id = tokenizer_config['bos_token_id']
        self.eos_token_id = tokenizer_config['eos_token_id']
        self.query_token_id = tokenizer_config['query_token_id']
        self.answer_token_id = tokenizer_config['answer_token_id']
        self.pad_token_id = tokenizer_config['pad_token_id']
        self.tokenizer = tokenizer

        if self.tokenizer is not None:
            self.bos_token, self.query_token, self.answer_token, self.eos_token = self.tokenizer.convert_ids_to_tokens(
                [self.bos_token_id, self.query_token_id, self.answer_token_id, self.eos_token_id]
            )
        else:
            self.bos_token, self.query_token, self.answer_token, self.eos_token = None, None, None, None

    def set_tokenizer(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.tokenizer = tokenizer
        self.bos_token, self.query_token, self.answer_token, self.eos_token = self.tokenizer.convert_ids_to_tokens(
            [self.bos_token_id, self.query_token_id, self.answer_token_id, self.eos_token_id]
        )

    def load_pretrained_weights(
            self, decoder: RxTDecoderComponent, encoder: RxTEncoderComponent,
            memory_attention: RxTInterlayerMemoryAttentionComponent
    ):
        self.decoder.load_state_dict(decoder.state_dict(), assign=True)
        self.encoder.load_state_dict(encoder.state_dict(), assign=True)
        self.memory_attention.load_state_dict(memory_attention.state_dict(), assign=True)

    def share_components(self):
        # 1. Load shared embeddings from encoder to decoder
        self.decoder.load_shared_embedding(self.encoder.model.embedding)
        # 2. Load shared STM from memory attention to decoder
        self.decoder.load_shared_memory(self.memory_attention.model.stm)
        # 3. Ensure correct initial memory shape
        self.set_batch_mode(self.batch_size != 1, self.batch_size)

    def set_batch_mode(self, use_batch_mode: bool, batch_size: Optional[int] = None):
        if use_batch_mode:
            self.memory_attention.model.stm.batched_memory(batch_size=batch_size, init_type='standard')
            self.batch_size = batch_size
        else:
            self.memory_attention.model.stm.single_memory(init_type='standard')
            self.batch_size = 1

    def prepare_stm_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.decoder.model.prepare_stm_kv_cache()

    def reset_self_attn_cache(self):
        return self.decoder.model.reset_self_attn_cache()

    def init_stm_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, add_noise: float = 0.0, force: bool = False):
        _, ed = self.encoder(input_ids, attention_mask=attention_mask)
        new_stm = ed if add_noise == 0.0 else ed + torch.randn_like(ed) * add_noise
        if force:
            self.memory_attention.model.stm.update_all(new_stm)
        else:
            self.memory_attention(new_stm, attention_mask=attention_mask)

    def reset_stm_state(self):
        self.memory_attention.model.stm.reset()

    def export_stm_state(self) -> torch.Tensor:
        return self.memory_attention.model.stm.memory.clone().detach()

    def load_stm_state(self, stm_state: torch.Tensor):
        device = self.memory_attention.model.stm.memory.device
        self.memory_attention.model.stm.update_all(stm_state.to(device))

    def tokenize_query(self, text: str, max_seq_len: int = 256, device: torch.device = torch.device("cpu")):
        tokenized = self.tokenizer(
            f'{self.bos_token}{self.query_token}{text}',
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=False
        )

        return {
            'input_ids': tokenized['input_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device)
        }

    def tokenize_batch(self, texts: list[str], max_seq_len: int = 256, device: torch.device = torch.device("cpu")):
        tokenized = self.tokenizer(
            [f'{self.bos_token}{self.query_token}{txt}' for txt in texts],
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=False
        )

        return {
            'input_ids': tokenized['input_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device)
        }

    def tokenize_full_interaction(self, query: str, answer: str, max_seq_len: int = 256, device: torch.device = torch.device("cpu")):
        tokenized = self.tokenizer(
            f'{self.bos_token}{self.query_token}{query}{self.answer_token}{answer}{self.eos_token}',
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=False
        )

        return {
            'input_ids': tokenized['input_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device)
        }

    def stringify_token(self, token_id: int, skip_special_tokens: bool = True, show_memory_update: bool = False) -> str:
        if token_id == -1:
            return '\n [STM update]' if show_memory_update else ''
        elif token_id == -2:
            return '\n [STM updated]' if show_memory_update else ''
        else:
            return decode_post_process(self.tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens))

    def stringify_tokens(self, generated_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        decoded = []
        for token_id in generated_ids:
            # Trim after end token
            decoded.append(decode_post_process(self.tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)))

        return decoded

    def stringify_batch(self, generated_ids: torch.Tensor) -> list[str]:
        decoded = []
        for seq in generated_ids:
            # Trim after end token
            end_pos = (seq == self.sampler.eos_token_id).nonzero()
            if end_pos.size(0) > 0:
                seq = seq[:end_pos[0] + 1]
            decoded.append(decode_post_process(self.tokenizer.decode(seq)))

        return decoded

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None,
            use_self_attn_cache: bool = False,
            current_positions: Optional[torch.Tensor] = None,
            action: RxTForwardAction = RxTForwardAction.DECODE
    ) -> torch.Tensor:
        if action == RxTForwardAction.DECODE:
            return self.decoder(input_ids, attention_mask=attention_mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)
        else:
            _, ed = self.encoder(input_ids, attention_mask=attention_mask)
            return self.memory_attention(ed, attention_mask=attention_mask)

    def _generate_single_token(
            self,
            input_ids: torch.Tensor,
            temperature: float,
            top_k: int,
            top_p: float,
            attention_mask: torch.Tensor,
            stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None,
            use_self_attn_cache: bool = True,
            init_step: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        device = input_ids.device

        # Forward pass to get next token logits
        outputs = self.forward(
            input_ids[:, -1].unsqueeze(0) if not init_step else input_ids, attention_mask=attention_mask[:, -1].unsqueeze(0) if not init_step else attention_mask, stm_kv_cache=stm_kv_cache,
            use_self_attn_cache=use_self_attn_cache, action=RxTForwardAction.DECODE, current_positions=torch.tensor([[input_ids.size(-1)]]).to(input_ids.device) if not init_step else None
        )
        next_token_logits = outputs[:, -1, :]  # Get logits for next token
        # Apply sampling
        next_token = sample(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        next_token = next_token.item()  # Extract scalar token
        next_token_ten = torch.tensor([[next_token]], device=device)
        next_input_ids = torch.cat([input_ids, next_token_ten], dim=1)
        new_one = torch.ones(1, 1, dtype=torch.bool, device=device)
        next_mask = torch.cat([attention_mask, new_one], dim=1) if attention_mask is not None else None

        # Yield the generated token
        return (
            next_token,
            next_input_ids,
            next_mask
        )

    def interact(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_seq_len: int = 256,
            use_self_attn_cache: bool = True,
    ) -> Iterator[int]:
        assert self.batch_size == 1 and input_ids.size(0) == 1, 'Batch size must be 1 in single interaction mode'

        with torch.no_grad():
            self.reset_self_attn_cache()

            stm_kv_cache = self.prepare_stm_kv_cache()

            input_ids = torch.cat([input_ids, torch.tensor([[self.answer_token_id]]).to(input_ids.device)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=attention_mask.device)], dim=-1)

            for i in range(max_seq_len - input_ids.size(-1)):
                next_token, input_ids, attention_mask = self._generate_single_token(
                    input_ids, temperature, top_k, top_p, attention_mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, init_step=i == 0 or not use_self_attn_cache
                )
                yield next_token
                if next_token == self.eos_token_id:
                    break

            yield -1 # start memory update
            self.forward(input_ids, attention_mask=attention_mask, action=RxTForwardAction.UPDATE) # input_ids and attention_mask are already accumulated
            yield -2 # finished memory update

    def batch_interact(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_seq_len: int = 256,
            use_self_attn_cache: bool = True,
    ) -> Iterator[torch.Tensor]:
        batch_size = input_ids.size(0)
        device = input_ids.device

        assert self.batch_size == batch_size, 'Input batch size must be the same as model (STM) batch size'

        initial_lens = attention_mask.sum(dim=1)
        for i in range(batch_size):
            input_ids[i, initial_lens[i]] = self.answer_token_id
            attention_mask[i, initial_lens[i]] = 1

        initial_lens += 1
        current_lens = initial_lens.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        working_ids = input_ids.clone()
        working_mask = attention_mask.clone()

        with torch.no_grad():
            stm_kv_cache = self.prepare_stm_kv_cache()

            self.reset_self_attn_cache()

            for step in range(max_seq_len):
                active = (~finished) & (current_lens < max_seq_len)
                if not active.any():
                    break

                max_len = current_lens.max().item()

                indices = (current_lens - 1).to(device)

                if step == 0 or not use_self_attn_cache:
                    # Slice input and mask up to the current max length among active sequences
                    inputs = working_ids[:, :max_len]
                    masks = working_mask[:, :max_len]

                    logits = self.forward(
                        inputs, attention_mask=masks, action=RxTForwardAction.DECODE,
                        stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache,
                        current_positions=None
                    )
                else:
                    finished_tokens = finished.unsqueeze(-1)
                    row_idx = torch.arange(working_ids.size(0))

                    selected_ids = working_ids[row_idx, indices].unsqueeze(-1)
                    selected_masks = working_mask[row_idx, indices].unsqueeze(-1)

                    inputs = torch.where(finished_tokens == 0, selected_ids, self.pad_token_id)
                    masks = torch.where(finished_tokens == 0, selected_masks, self.pad_token_id)

                    logits = self.forward(
                        inputs, attention_mask=masks, action=RxTForwardAction.DECODE,
                        stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache,
                        current_positions=indices
                    )

                # Get the last valid token index for each active sequence
                if step == 0 or not use_self_attn_cache:
                    last_logits = logits[torch.arange(batch_size, device=logits.device), indices]
                else:
                    last_logits = logits[:, -1]

                # Sample next tokens and log probs
                next_tokens, _ = sample_batch(
                    last_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Prepare active sequences mask
                active_mask = (~finished) & (current_lens < max_seq_len)
                if not active_mask.any():
                    break
                # Get positions to update
                positions_to_update = current_lens
                # Prepare indexing batch range
                batch_range = torch.arange(batch_size, device=self.device)
                # Vectorized working tensors update
                working_ids[batch_range[active_mask], positions_to_update[active_mask]] = next_tokens[active_mask]
                working_mask[batch_range[active_mask], positions_to_update[active_mask]] = 1
                # Update lens for active tokens
                current_lens += active_mask.long()
                # Update finished tensor if some batch items stopped generation
                finished |= (next_tokens == self.end_token_id) & active_mask

                yield working_ids[torch.arange(batch_size), current_lens - 1].squeeze(-1)

            yield torch.full((batch_size,), -1, dtype=torch.long) # start memory update
            # Update memory
            self.forward(working_ids, attention_mask=working_mask, action=RxTForwardAction.UPDATE)
            yield torch.full((batch_size,), -2, dtype=torch.long) # finished memory update


    def batch_interactions(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_seq_len: int = 256,
            use_self_attn_cache: bool = True,
            return_interactions_with_queries: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.size(0)
        device = input_ids.device

        assert self.batch_size == batch_size, 'Input batch size must be the same as model (STM) batch size'

        initial_lens = attention_mask.sum(dim=1)
        for i in range(batch_size):
            input_ids[i, initial_lens[i]] = self.answer_token_id
            attention_mask[i, initial_lens[i]] = 1

        initial_lens += 1
        current_lens = initial_lens.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        working_ids = input_ids.clone()
        working_mask = attention_mask.clone()

        with torch.no_grad():
            stm_kv_cache = self.prepare_stm_kv_cache()

            self.reset_self_attn_cache()

            for step in range(max_seq_len):
                active = (~finished) & (current_lens < max_seq_len)
                if not active.any():
                    break

                max_len = current_lens.max().item()

                indices = (current_lens - 1).to(device)

                if step == 0 or not use_self_attn_cache:
                    # Slice input and mask up to the current max length among active sequences
                    inputs = working_ids[:, :max_len]
                    masks = working_mask[:, :max_len]

                    logits = self.forward(
                        inputs, attention_mask=masks, action=RxTForwardAction.DECODE,
                        stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache,
                        current_positions=None
                    )
                else:
                    finished_tokens = finished.unsqueeze(-1)
                    row_idx = torch.arange(working_ids.size(0))

                    selected_ids = working_ids[row_idx, indices].unsqueeze(-1)
                    selected_masks = working_mask[row_idx, indices].unsqueeze(-1)

                    inputs = torch.where(finished_tokens == 0, selected_ids, self.pad_token_id)
                    masks = torch.where(finished_tokens == 0, selected_masks, self.pad_token_id)

                    logits = self.forward(
                        inputs, attention_mask=masks, action=RxTForwardAction.DECODE,
                        stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache,
                        current_positions=indices
                    )

                # Get the last valid token index for each active sequence
                if step == 0 or not use_self_attn_cache:
                    last_logits = logits[torch.arange(batch_size, device=logits.device), indices]
                else:
                    last_logits = logits[:, -1]

                # Sample next tokens and log probs
                next_tokens, _ = sample_batch(
                    last_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Prepare active sequences mask
                active_mask = (~finished) & (current_lens < max_seq_len)
                if not active_mask.any():
                    break
                # Get positions to update
                positions_to_update = current_lens
                # Prepare indexing batch range
                batch_range = torch.arange(batch_size, device=self.device)
                # Vectorized working tensors update
                working_ids[batch_range[active_mask], positions_to_update[active_mask]] = next_tokens[active_mask]
                working_mask[batch_range[active_mask], positions_to_update[active_mask]] = 1
                # Update lens for active tokens
                current_lens += active_mask.long()
                # Update finished tensor if some batch items stopped generation
                finished |= (next_tokens == self.end_token_id) & active_mask

            # Update memory
            self.forward(working_ids, attention_mask=working_mask, action=RxTForwardAction.UPDATE)

            if return_interactions_with_queries:
                return working_ids, working_mask
            else:
                # Extract generated tokens
                generated_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
                generated_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
                for i in range(batch_size):
                    start = initial_lens[i].item()
                    end = current_lens[i].item()
                    gen_len = min(end - start + 1, max_seq_len) # +1 for added [A] token
                    if gen_len > 0:
                        generated_ids[i, :gen_len] = working_ids[i, start-1:end] # -1 to include [A] token
                        generated_mask[i, :gen_len] = working_mask[i, start-1:end] # -1 to include [A] token

                return generated_ids, generated_mask


class RxTBeta(RxTAlpha):
    def __init__(
            self,
            decoder_config: RxTComponentConfig,
            encoder_config: RxTComponentConfig,
            memory_attention_config: RxTInterlayerMemoryAttentionConfig,
            tokenizer_config: RxTAlphaTokenizerConfig,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            **kwargs,
    ):
        super(RxTBeta, self).__init__(
            decoder_config=decoder_config,
            encoder_config=encoder_config,
            memory_attention_config=memory_attention_config,
            tokenizer_config=tokenizer_config,
            tokenizer=tokenizer,
            memory_attention_variant='self-interlayer',
            **kwargs
        )
