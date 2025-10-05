import torch
import torch.nn as nn
from enum import Enum
from typing import TypeAlias, Union
from ..rxt.models import (
    RxTDecoder, RxTEncoder, RxTSimpleMemoryAttention,
    RxTSelfMemoryAttention, RxTInterlayerMemoryAttention, RxTSelfInterlayerMemoryAttention
)


RxTMemoryAttentionType: TypeAlias = Union[
    RxTSimpleMemoryAttention, RxTSelfMemoryAttention,
    RxTInterlayerMemoryAttention, RxTSelfInterlayerMemoryAttention
]


class RxTMrqBenchAction(Enum):
    DECODE = 1
    UPDATE = 2


class RxTMrqBenchModel(nn.Module):
    def __init__(
            self,
            encoder: RxTEncoder,
            decoder: RxTDecoder,
            memory_attention: RxTMemoryAttentionType,
            **kwargs
    ):
        super(RxTMrqBenchModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.memory_attention = memory_attention

    def reset_memory(self):
        self.memory_attention.reset_memory()

    def clone_reset_memory(self):
        self.memory_attention.clone_reset_memory()

    def unique_parameters(self, with_embedding: bool = True):
        if with_embedding:
            return list(set(
                list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.memory_attention.parameters())
            ))
        else:
            return list(set(
                self.not_memory_parameters() +
                self.memory_cross_attention_parameters() +
                list(self.memory_attention_parameters())
            ))

    def prepare_stm_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.decoder.model.prepare_stm_kv_cache()

    def reset_self_attn_cache(self):
        return self.decoder.model.reset_self_attn_cache()

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            stm_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None,
            use_self_attn_cache: bool = False,
            current_positions: torch.Tensor = None,
            action: RxTMrqBenchAction = RxTMrqBenchAction.DECODE
    ) -> torch.Tensor:
        if action == RxTMrqBenchAction.DECODE:
            return self.decoder(x, attention_mask=attention_mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)
        else:
            _, ed = self.encoder(x, attention_mask=attention_mask)
            return self.memory_attention(ed, attention_mask=attention_mask)
