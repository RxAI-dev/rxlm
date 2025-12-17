import torch
from torch import nn
from typing import TypedDict, Optional, Iterator, Union, Literal
from huggingface_hub import PyTorchModelHubMixin
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..training.tokenizer import decode_post_process
from ..transformers.positional import RotaryPositionalEmbedding
from ..transformers.attention import init_attention
from ..transformers.llm_layers import ClassicTransformerLayer
from ..transformers.llm_models import ClassicTransformerDecoder
from ..transformers.ff import get_activation_layer
from ..transformers.sampler import sample
from ..utils import get_model_size
from ..experimental.attention import init_experimental_attention


class DecoderOnlyTransformerConfig(TypedDict):
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
    ff_num_shared_experts: int
    att_type: str
    att_num_experts: int
    att_num_query_experts: int
    att_num_query_groups: int
    att_num_global_tokens: int
    att_window_size: int
    use_head_norm: bool
    init_identity_norm: bool
    tie_embeddings: bool
    head_norm_type: str


class DecoderOnlyTransformer(nn.Module, PyTorchModelHubMixin, pipeline_tag="text-generation", license="apache-2.0"):
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
            use_flash_attention: bool = False,
            use_gated: bool = True,
            ff_activation: str = "swish",
            ff_dropout: float = 0.0,
            att_dropout: float = 0.0,
            use_rms_norm: bool = True,
            att_groups: int = 1,
            use_moe_ff: bool = False,
            ff_num_experts: int = 1,
            ff_moe_top_k: int = 1,
            ff_num_shared_experts: int = 0,
            num_initial_dense_layers: int = 0,
            dense_ff_dim: int = 384,
            att_type: str = 'sqa',
            att_num_experts: int = None,
            att_num_query_experts: int = None,
            att_num_query_groups: int = None,
            att_num_global_tokens: int = 16,
            att_window_size: int = 128,
            use_head_norm: bool = False,
            init_identity_norm: bool = False,
            tie_embeddings: bool = False,
            head_norm_type: str = 'layer_norm',
            router_amp: bool = False,
            router_dtype: torch.dtype = torch.float32,
            use_vectorized_moe: bool = True,
            vectorized_moe_from_legacy: bool = False,
            moe_grouped_gemm: bool = True,
            moe_bias_mode: Literal['global', 'local', 'off'] = 'global',
            moe_shared_experts_bias_mode: Literal['global', 'local', 'off'] = 'local',
            moe_use_weighted_shared_experts: bool = False,
            use_gated_attention: bool = False,
            gated_attention_activation: str = 'sigmoid',
            use_attention_output_bias: bool = True,  # legacy compat
            moe_use_cutlass_grouped_gemm: bool = True,
            **kwargs
    ):
        super(DecoderOnlyTransformer, self).__init__(**kwargs)
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
                                              max_seq_len=seq_len, is_causal=True, num_query_groups=att_num_query_groups,
                                              use_gated_attention=use_gated_attention, gated_attention_activation=gated_attention_activation,
                                              use_output_bias=use_attention_output_bias)
        else:
            att_init = lambda: init_experimental_attention(embed_dim, att_heads, att_type, att_groups, rope=rope,
                                                           use_flash_attention=use_flash_attention, dropout=att_dropout,
                                                           max_seq_len=seq_len, is_causal=True, num_experts=att_num_experts,
                                                           num_query_experts=att_num_query_experts, num_query_groups=att_num_query_groups,
                                                           num_global_tokens=att_num_global_tokens, window_size=att_window_size)

        use_moe_att = att_type in ['gma', 'dma']

        def layer_init(i: int):
            if i < num_initial_dense_layers:
                return ClassicTransformerLayer(
                    embed_dim,
                    dense_ff_dim,
                    use_gated=use_gated,
                    use_moe=False,
                    ff_activation=ff_activation,
                    ff_dropout=ff_dropout,
                    use_rms_norm=use_rms_norm,
                    self_attention=att_init(),
                    use_moe_att=use_moe_att,
                    )
            else:
                return ClassicTransformerLayer(
                    embed_dim,
                    ff_dim,
                    use_gated=use_gated,
                    use_moe=use_moe_ff,
                    num_experts=ff_num_experts,
                    num_shared_experts=ff_num_shared_experts,
                    moe_top_k=ff_moe_top_k,
                    ff_activation=ff_activation,
                    ff_dropout=ff_dropout,
                    use_rms_norm=use_rms_norm,
                    self_attention=att_init(),
                    use_moe_att=use_moe_att,
                    router_amp=router_amp,
                    router_dtype=router_dtype,
                    use_vectorized_moe=use_vectorized_moe,
                    vectorized_moe_from_legacy=vectorized_moe_from_legacy,
                    moe_grouped_gemm=moe_grouped_gemm,
                    moe_bias_mode=moe_bias_mode,
                    moe_shared_experts_bias_mode=moe_shared_experts_bias_mode,
                    moe_use_weighted_shared_experts=moe_use_weighted_shared_experts,
                    moe_use_cutlass_grouped_gemm=moe_use_cutlass_grouped_gemm,
                )


        self.model = ClassicTransformerDecoder(
            embed_dim,
            vocab_size,
            embedding=embedding,
            layers=nn.ModuleList([layer_init(i) for i in range(num_layers)]),
            use_flash_attention=use_flash_attention,
            use_head_norm=use_head_norm,
            init_identity_norm=init_identity_norm,
            tie_embeddings=tie_embeddings,
            head_norm_type=head_norm_type,
        )

    def params_count(self):
        return get_model_size(self.model)

    def reset_self_attn_cache(self):
        return self.model.reset_self_attn_cache()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, use_self_attn_cache: bool = False, current_positions: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask, use_self_attn_cache=use_self_attn_cache, current_positions=current_positions)

    def _generate_single_token(
            self,
            input_ids: torch.Tensor,
            temperature: float,
            top_k: int,
            top_p: float,
            attention_mask: torch.Tensor,
            use_self_attn_cache: bool = True,
            init_step: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        device = input_ids.device

        # Forward pass to get next token logits
        outputs = self.forward(
            input_ids[:, -1].unsqueeze(0) if not init_step else input_ids, attention_mask=attention_mask[:, -1].unsqueeze(0) if not init_step else attention_mask,
            use_self_attn_cache=use_self_attn_cache, current_positions=torch.tensor([[input_ids.size(-1)]]).to(input_ids.device) if not init_step else None
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

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_seq_len: int = 256,
            use_self_attn_cache: bool = True,
            answer_token_id: Optional[int] = 6,
            eos_token_id: Optional[int] = 3,
    ) -> Iterator[int]:
        assert input_ids.size(0) == 1, 'Batch size must be 1 in single interaction mode'

        with torch.no_grad():
            self.reset_self_attn_cache()

            input_ids = torch.cat([input_ids, torch.tensor([[answer_token_id]]).to(input_ids.device)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=attention_mask.device)], dim=-1)

            for i in range(max_seq_len - input_ids.size(-1)):
                next_token, input_ids, attention_mask = self._generate_single_token(
                    input_ids, temperature, top_k, top_p, attention_mask,
                    use_self_attn_cache=use_self_attn_cache, init_step=i == 0 or not use_self_attn_cache
                )
                yield next_token
                if next_token == eos_token_id:
                    break

    def stringify_token(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], token_id: int, skip_special_tokens: bool = True) -> str:
        return decode_post_process(tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens))

    def _build_full_conversation_text(self, chat_history: list[list[str]], new_query: str, tokenizer_config: dict) -> str:
        full_text = f"{tokenizer_config['bos_token']}"

        for query, answer in chat_history:
            full_text += f"{tokenizer_config['query_token']}{query}{tokenizer_config['answer_token']}{answer}"

        full_text += f"{tokenizer_config['query_token']}{new_query}"
        return full_text

    def tokenize_chat_template(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            chat_history: list[Union[dict[Literal['role', 'content'], str], list[str]]],
            new_query: str,
            max_seq_len: int = 256,
            use_simplified_format: bool = False,
            tokenized_config: dict = None
    ) -> dict[str, torch.Tensor]:
            if not use_simplified_format:
                def get_simplified_item(idx: int) -> list[str]:
                    query = chat_history[idx]
                    assert query['role'] == 'user'
                    answer = chat_history[idx + 1]
                    assert answer['role'] == 'user'
                    return [query['content'], answer['content']]
                chat_history = [get_simplified_item(i) for i in range(0, len(chat_history), 2)]

            if tokenized_config is None:
                tokenized_config = {
                    'bos_token': '[BOS]',
                    'eos_token': '[EOS]',
                    'query_token': '[Q]',
                    'answer_token': '[A]',
                }

            full_text = self._build_full_conversation_text(chat_history, new_query, tokenized_config)

            enc = tokenizer(
                full_text,
                max_length=max_seq_len,
                truncation=True,
                padding=False,
                return_tensors='pt',
                return_attention_mask=True,
                add_special_tokens=False
            )

            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }