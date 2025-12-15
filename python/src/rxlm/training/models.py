import torch
import torch.nn as nn
from enum import Enum
from typing import Literal, Iterator, Optional, TypeAlias, Union
from huggingface_hub import PyTorchModelHubMixin
from ..transformers.ff import GatedLinearUnit, get_activation_layer
from ..rxt.models import (
    RxTDecoder, RxTEncoder, RxTSimpleMemoryAttention,
    RxTSelfMemoryAttention, RxTInterlayerMemoryAttention, RxTSelfInterlayerMemoryAttention
)

RxTMemoryAttentionType: TypeAlias = Union[
    RxTSimpleMemoryAttention, RxTSelfMemoryAttention,
    RxTInterlayerMemoryAttention, RxTSelfInterlayerMemoryAttention
]

class MLMHead(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    def __init__(self, embed_dim: int, vocab_size: int, *args, **kwargs):
        super(MLMHead, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class JointTrainingModel(nn.Module):
    def __init__(
            self,
            encoder: RxTEncoder,
            decoder: RxTDecoder,
            mlm_head: MLMHead,
            noise_level: float = None,
            masking_prob: float = None,
            *args,
            **kwargs
    ):
        super(JointTrainingModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.mlm_head = mlm_head
        self.decoder = decoder

        self.noise_level = noise_level
        self.masking_prob = masking_prob

    def forward_one_result(self, x_e: torch.Tensor, x_d: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        self.decoder.model.stm.reset()

        _, encoded_layers = self.encoder(x_e, attention_mask=attention_mask)
        self.decoder.model.stm.update_all(encoded_layers)
        y_d = self.decoder(x_d, attention_mask=attention_mask)
        return y_d

    def forward(self, x_e: torch.Tensor, x_d: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor]:

        self.decoder.model.stm.reset()

        encoder_result, encoded_layers = self.encoder(x_e, attention_mask=attention_mask)
        y_e = self.mlm_head(encoder_result)

        # Optimized: detach without clone - we're not modifying the original
        with torch.no_grad():
            fake_stm = encoded_layers.detach()

            # Apply masking in-place style operations where possible
            if self.masking_prob is not None:
                # Use bernoulli_ for in-place generation where applicable
                mask_shape = fake_stm.shape[:-1]
                results_mask = torch.empty(mask_shape, device=fake_stm.device, dtype=fake_stm.dtype)
                results_mask.bernoulli_(1.0 - self.masking_prob)
                fake_stm = fake_stm * results_mask.unsqueeze(-1)

            # Add noise if needed - fused operation
            if self.noise_level is not None:
                noise = torch.empty_like(fake_stm).normal_(0, self.noise_level)
                fake_stm = fake_stm + noise

        self.decoder.model.stm.update_all(fake_stm)
        y_d = self.decoder(x_d, attention_mask=attention_mask)
        return y_e, y_d


class MemoryAttentionTrainingModel(nn.Module):
    def __init__(
            self,
            encoder: RxTEncoder,
            memory_attention: RxTMemoryAttentionType,
            *args,
            **kwargs
    ):
        super(MemoryAttentionTrainingModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.memory_attention = memory_attention

    def trainable_parameters(self) -> Iterator[torch.Tensor]:
        return self.memory_attention.parameters()

    def reset_memory(self, init_type: str = None):
        self.memory_attention.reset_memory(init_type)

    def clone_reset_memory(self):
        self.memory_attention.clone_reset_memory()

    def get_memory_state(self):
        return self.memory_attention.model.stm.memory

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, noise_level: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, encoded_layers = self.encoder(input_ids, attention_mask=attention_mask)
            noisy_encoded_data = encoded_layers + noise_level * torch.randn_like(encoded_layers)

        new_stm = self.memory_attention(noisy_encoded_data, attention_mask=attention_mask)
        return new_stm, encoded_layers


class SupervisedMemoryAwareModel(nn.Module):
    def __init__(
            self,
            encoder: RxTEncoder,
            decoder: RxTDecoder,
            memory_attention: RxTMemoryAttentionType,
            train_only_decoder: bool = False,
            *args,
            **kwargs
    ):
        super(SupervisedMemoryAwareModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.memory_attention = memory_attention
        self.train_only_decoder = train_only_decoder

    def trainable_parameters(self) -> Iterator[torch.Tensor]:
        if self.train_only_decoder:
            return self.decoder.parameters()
        else:
            params_set = set(
                list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.memory_attention.parameters())
            )
            return iter(params_set)

    def clone_reset_memory(self):
        self.memory_attention.clone_reset_memory()

    def reset_memory(self, init_type: str = None):
        self.memory_attention.reset_memory(init_type)

    def forward(self, x_e: torch.Tensor, x_d: torch.Tensor, encoder_mask: torch.Tensor = None, decoder_mask: torch.Tensor = None, is_first_step: bool = False) -> torch.Tensor:
        if not is_first_step:
            with torch.set_grad_enabled(not self.train_only_decoder):
                _, encoded_layers = self.encoder(x_e, attention_mask=encoder_mask)
                self.memory_attention(encoded_layers, attention_mask=encoder_mask)

        logits = self.decoder(x_d, attention_mask=decoder_mask)
        return logits


class MrlActorAction(Enum):
    DECODE = 1
    UPDATE = 2


class MrlActorModel(nn.Module):
    def __init__(
            self,
            encoder: RxTEncoder,
            decoder: RxTDecoder,
            memory_attention: RxTMemoryAttentionType,
            **kwargs
    ):
        super(MrlActorModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.memory_attention = memory_attention

    def freeze_components(self, stage: Literal['warmup', 'update', 'fetch'] = 'fetch', freeze_embeddings: bool = False):
        """Freeze encoder/decoder except memory-related layers."""
        # Freeze/unfreeze encoder
        if self.encoder.freeze_all is not None:
            if stage == 'update':
                self.encoder.unfreeze_all()
            else:
                self.encoder.freeze_all()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True if stage == 'update' else False

        # Freeze/unfreeze decoder
        if self.decoder.freeze_without_memory is not None:
            if stage == 'fetch' or stage == 'update':
                self.decoder.freeze_without_memory(unfreeze_norms=True)
            else:
                self.decoder.freeze_without_memory(unfreeze_norms=True)
                self.decoder.freeze_memory(with_norms=True)
        else:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.model.trainable_cross_attention_(True if stage == 'fetch' or stage == 'update' else False, with_norms=True)

        # Freeze/unfreeze memory attention
        if self.memory_attention.unfreeze is not None:
            self.memory_attention.unfreeze()
        else:
            for param in self.memory_attention.parameters():
                param.requires_grad = True

        if freeze_embeddings:
            for param in self.encoder.model.embedding.parameters():
                param.requires_grad = False

    def unfreeze_components(self, freeze_embeddings: bool = False):
        """Unfreeze all components after initial training."""
        if self.encoder.unfreeze_all is not None:
            self.encoder.unfreeze_all()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

        if self.decoder.unfreeze_all is not None:
            self.decoder.unfreeze_all()
        else:
            for param in self.decoder.parameters():
                param.requires_grad = True

        if self.memory_attention.unfreeze is not None:
            self.memory_attention.unfreeze()
        else:
            for param in self.memory_attention.parameters():
                param.requires_grad = True

        if freeze_embeddings:
            for param in self.encoder.model.embedding.parameters():
                param.requires_grad = False


    def reset_memory(self):
        self.memory_attention.reset_memory()

    def clone_reset_memory(self):
        self.memory_attention.clone_reset_memory()

    def memory_parameters(self) -> list[nn.Parameter]:
        return list(set(
            self.encoder.memory_parameters() +
            self.decoder.memory_parameters() +
            list(self.memory_attention.parameters())
        ))

    def memory_cross_attention_parameters(self) -> list[nn.Parameter]:
        return list(set(
            self.encoder.memory_parameters() +
            self.decoder.memory_parameters()
        ))

    def memory_attention_parameters(self) -> Iterator[nn.Parameter]:
        return self.memory_attention.parameters()

    def not_memory_parameters(self) -> list[nn.Parameter]:
        return list(set(
            self.encoder.not_memory_parameters() +
            self.decoder.not_memory_parameters()
        ))

    def embedding_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.model.embedding.parameters()

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

    def moe_router_loss(self) -> Optional[torch.Tensor]:
        if self.encoder.model.use_moe and self.decoder.model.use_moe:
            return (self.encoder.model.moe_router_loss() + self.decoder.model.moe_router_loss()) / 2
        elif self.encoder.model.use_moe:
            return self.encoder.model.moe_router_loss()
        elif self.decoder.model.use_moe:
            return self.decoder.model.moe_router_loss()
        else:
            return None

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
            action: MrlActorAction = MrlActorAction.DECODE
    ) -> torch.Tensor:
        if action == MrlActorAction.DECODE:
            return self.decoder(x, attention_mask=attention_mask, stm_kv_cache=stm_kv_cache, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)
        else:
            _, ed = self.encoder(x, attention_mask=attention_mask)
            return self.memory_attention(ed, attention_mask=attention_mask)


class MrlCriticModel(nn.Module, PyTorchModelHubMixin, license="apache-2.0", pipeline_tag="text-classification"):
    def __init__(self, encoder: nn.Module, embed_dim: int, init_scale: float = 10.0, activation: Literal['sigmoid', 'tanh', 'linear'] = 'sigmoid', **kwargs):
        super(MrlCriticModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.value_head = nn.Sequential(
            GatedLinearUnit(embed_dim, embed_dim, nn.SiLU()),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            get_activation_layer(activation)
        )
        # Learnable scaling parameters
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def head_parameters(self) -> list[nn.Parameter]:
        return [*list(self.value_head.parameters()), self.scale, self.shift]

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x, _ = self.encoder(x, attention_mask=attention_mask)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)

        return self.value_head(x) * self.scale + self.shift
