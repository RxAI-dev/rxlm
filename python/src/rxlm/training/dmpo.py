"""
Direct Memory and Preference Optimization (DMPO) Training Module

This module implements DMPO, a DPO-like algorithm adapted for Reactive Transformer's
memory-aware training. DMPO extends SMAT (Supervised Memory-Aware Training) by incorporating
preference learning through accepted/rejected answer pairs.

The key innovations of DMPO:
1. DPO-style preference learning with memory context
2. Memory updates based only on accepted interactions
3. Reference model for stable preference optimization
4. Optional anchored variant (APO-style) with neutral answers

Theoretical Background:
-----------------------

**Direct Preference Optimization (DPO)** [Rafailov et al., 2023] provides a simpler alternative
to RLHF by directly optimizing the policy to prefer accepted over rejected responses. The standard
DPO loss is:

    L_DPO = -E[log σ(β (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]

where:
    - y_w: accepted (preferred) response
    - y_l: rejected (dispreferred) response
    - π: current policy
    - π_ref: frozen reference policy
    - β: temperature parameter controlling preference strength

**DMPO Extension for Reactive Transformer:**

DMPO adapts DPO for the memory-aware, multi-step setting of RxT:

1. **Memory-Conditioned Preferences**: At each interaction step t, the model generates
   responses conditioned on the accumulated memory state STM_{t-1}. Preferences are
   learned in this memory-conditioned context.

2. **Accepted-Only Memory Updates**: Unlike standard SMAT where all interactions update
   memory, DMPO only updates memory based on accepted interactions. This ensures the
   memory system learns to store high-quality information.

3. **Multi-Step Preference Propagation**: Preference learning propagates through the
   memory update chain, allowing the model to learn which memory states lead to better
   downstream responses.

**Anchored Preference Optimization (APO) Extension:**

The anchored variant adds a neutral reference point, creating a three-way preference:
    - Accepted: Strongly preferred response
    - Neutral: Reference baseline (neither good nor bad)
    - Rejected: Dispreferred response

The anchored loss combines standard DPO with regularization toward the neutral response:

    L_APO = L_DPO + λ * E[|log π(y_n|x) - log π_ref(y_n|x)|]

This prevents the model from drifting too far from reasonable behavior while still
learning strong preferences.

References:
-----------
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- Filipek, "Reactive Transformer: Stateful Real-Time Processing for Event-Driven AI" (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
import math
import copy
from typing import Union, Callable, Any, Optional, Iterator, Literal
from datasets import Dataset as HfDataset, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .base import BaseTrainer
from .models import SupervisedMemoryAwareModel, RxTMemoryAttentionType
from .ddp import distributed_value_mean, distributed_mean
from .utils import TokenizedDict, smart_concat
from .dataset import HybridReasoningSmatDataset
from ..rxt.models import RxTDecoder, RxTEncoder


class DmpoDataset(Dataset):
    """
    Dataset for Direct Memory and Preference Optimization (DMPO).

    This dataset extends HybridReasoningSmatDataset with preference data.
    Each interaction has:
    - Query part (same as SMAT)
    - Accepted answer part (preferred response)
    - Rejected answer part (dispreferred response)
    - Optional: Neutral answer part (for anchored variant)

    Dataset Format:
    ---------------
    Each episode should have an 'interactions' field containing a list of interactions.
    Each interaction has:
    - internal: Optional internal instruction
    - query: User query (or tool_use for tool usage)
    - tool_use: Tool result (alternative to query)
    - accepted: Dict with accepted answer fields (think, answer, tool_call)
    - rejected: Dict with rejected answer fields (think, answer, tool_call)
    - neutral: Optional dict with neutral answer fields (for anchored variant)
    """

    def __init__(
            self,
            episodes: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            # Field names
            interactions_field: str = 'interactions',
            query_field: str = 'query',
            answer_field: str = 'answer',
            think_field: str = 'think',
            tool_call_field: str = 'tool_call',
            tool_use_field: str = 'tool_use',
            internal_field: str = 'internal',
            accepted_field: str = 'accepted',
            rejected_field: str = 'rejected',
            neutral_field: str = 'neutral',  # For anchored variant
            # Special tokens
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            think_token: str = '[T]',
            tool_call_token: str = '[C]',
            tool_use_token: str = '[U]',
            internal_token: str = '[I]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            # Options
            use_anchored: bool = False,
            **kwargs,
    ):
        super(DmpoDataset, self).__init__(**kwargs)
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Field names
        self.interactions_field = interactions_field
        self.query_field = query_field
        self.answer_field = answer_field
        self.think_field = think_field
        self.tool_call_field = tool_call_field
        self.tool_use_field = tool_use_field
        self.internal_field = internal_field
        self.accepted_field = accepted_field
        self.rejected_field = rejected_field
        self.neutral_field = neutral_field

        # Special tokens
        self.query_token = query_token
        self.answer_token = answer_token
        self.think_token = think_token
        self.tool_call_token = tool_call_token
        self.tool_use_token = tool_use_token
        self.internal_token = internal_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.use_anchored = use_anchored
        self.is_pre_tokenized = False
        self.is_list = isinstance(self.episodes, list)
        self.inputs = []

    def __len__(self):
        return len(self.inputs if self.is_pre_tokenized else self.episodes)

    def _build_query_text(self, interaction: dict) -> str:
        """Build query part text: [BOS][I]internal[Q|U]query_or_tool_use"""
        parts = [self.bos_token]

        internal = interaction.get(self.internal_field)
        if internal:
            parts.append(f"{self.internal_token}{internal}")

        tool_use = interaction.get(self.tool_use_field)
        query = interaction.get(self.query_field)

        if tool_use:
            parts.append(f"{self.tool_use_token}{tool_use}")
        elif query:
            parts.append(f"{self.query_token}{query}")

        return ''.join(parts)

    def _build_answer_text(self, answer_data: dict) -> str:
        """Build answer part text: [T]think[A]answer[C]tool_call[EOS]"""
        parts = []

        think = answer_data.get(self.think_field)
        if think:
            parts.append(f"{self.think_token}{think}")

        answer = answer_data.get(self.answer_field)
        if answer:
            parts.append(f"{self.answer_token}{answer}")

        tool_call = answer_data.get(self.tool_call_field)
        if tool_call:
            parts.append(f"{self.tool_call_token}{tool_call}")

        parts.append(self.eos_token)
        return ''.join(parts)

    def _tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize text and return input_ids and attention_mask."""
        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,
        )

        input_ids = enc['input_ids'][0]
        if not (input_ids < self.tokenizer.vocab_size).all():
            input_ids[input_ids >= self.tokenizer.vocab_size] = self.tokenizer.unk_token_id
        if not (input_ids >= 0).all():
            input_ids[input_ids < 0] = self.tokenizer.unk_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': enc['attention_mask'][0],
        }

    def _tokenize_interaction(self, interaction: dict) -> dict:
        """
        Tokenizes a single interaction with preference data.

        Returns:
            Dict with:
            - query: Tokenized query part
            - accepted: Tokenized accepted answer
            - rejected: Tokenized rejected answer
            - neutral: Optional tokenized neutral answer (if anchored mode)
        """
        # Build and tokenize query part
        query_text = self._build_query_text(interaction)
        query_enc = self._tokenize_text(query_text)

        # Get answer data
        accepted_data = interaction.get(self.accepted_field, {})
        rejected_data = interaction.get(self.rejected_field, {})

        # Build and tokenize answer parts
        accepted_text = self._build_answer_text(accepted_data)
        rejected_text = self._build_answer_text(rejected_data)

        accepted_enc = self._tokenize_text(accepted_text)
        rejected_enc = self._tokenize_text(rejected_text)

        result = {
            'query': query_enc,
            'accepted': accepted_enc,
            'rejected': rejected_enc,
        }

        # Handle anchored variant
        if self.use_anchored:
            neutral_data = interaction.get(self.neutral_field, {})
            if neutral_data:
                neutral_text = self._build_answer_text(neutral_data)
                result['neutral'] = self._tokenize_text(neutral_text)
            else:
                # Use empty tensor as placeholder
                result['neutral'] = {
                    'input_ids': torch.zeros(self.max_seq_len, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_seq_len, dtype=torch.long),
                }

        return result

    def get_tokenized_item(self, idx: int, episode: dict = None) -> dict:
        if self.is_pre_tokenized:
            return self.inputs[idx]

        item = self.episodes[idx] if episode is None else episode
        interactions = item[self.interactions_field]

        # Tokenize all interactions
        tokenized_interactions = [
            self._tokenize_interaction(inter) for inter in interactions
        ]

        if tokenized_interactions:
            first = tokenized_interactions[0]
            follow_ups = tokenized_interactions[1:] if len(tokenized_interactions) > 1 else []

            return {
                'query': first['query'],
                'accepted': first['accepted'],
                'rejected': first['rejected'],
                **({'neutral': first['neutral']} if self.use_anchored else {}),
                'interactions': follow_ups,
            }
        else:
            # Empty episode
            empty = {
                'input_ids': torch.zeros(self.max_seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_seq_len, dtype=torch.long),
            }
            result = {
                'query': empty,
                'accepted': {k: v.clone() for k, v in empty.items()},
                'rejected': {k: v.clone() for k, v in empty.items()},
                'interactions': [],
            }
            if self.use_anchored:
                result['neutral'] = {k: v.clone() for k, v in empty.items()}
            return result

    def __getitem__(self, idx: int) -> dict:
        return self.get_tokenized_item(idx)

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000, keep_order: bool = False):
        """Pre-tokenizes all items for faster training."""
        if not self.is_pre_tokenized:
            num_episodes = len(self.episodes)
            eps = self.episodes if self.is_list else self.episodes.to_list()
            del self.episodes
            self.episodes = None
            for index in range(num_episodes):
                self.inputs.append(self.get_tokenized_item(
                    index,
                    episode=eps.pop() if not keep_order else eps[index]
                ))
                if verbose and index % log_interval == 0:
                    print(f'Processed {index + 1}/{num_episodes}')
            del eps
            self.is_pre_tokenized = True

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "DmpoDataset":
        split_point = int(len(self.inputs if self.is_pre_tokenized else self.episodes) * ((1 - size) if not from_start else size))
        if not isinstance(self.episodes, list):
            subset = self.episodes.select(
                range(split_point, len(self.episodes)) if not from_start else range(split_point))
            self.episodes = self.episodes.select(
                range(split_point) if not from_start else range(split_point, len(self.episodes)))
        else:
            subset = self.episodes[split_point:-1] if not from_start else self.episodes[0:split_point]
            self.episodes = self.episodes[0:split_point] if not from_start else self.episodes[split_point:-1]
        return self.__class__(
            subset,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            interactions_field=self.interactions_field,
            use_anchored=self.use_anchored,
            **kwargs
        )

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            subset: str = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            split: str = 'train',
            interactions_field: str = 'interactions',
            load_kwargs: dict = None,
            max_seq_len: int = 1024,
            use_anchored: bool = False,
            **kwargs
    ):
        """Load dataset from HuggingFace Hub."""
        if load_kwargs is None:
            load_kwargs = {}

        hf_dataset = load_dataset(dataset_id, subset, split=split, **load_kwargs) if subset else load_dataset(dataset_id, split=split, **load_kwargs)

        return cls(
            hf_dataset,
            tokenizer,
            interactions_field=interactions_field,
            max_seq_len=max_seq_len,
            use_anchored=use_anchored,
            **kwargs
        )

    @staticmethod
    def collate_dmpo_batch(batch: list[dict]) -> dict:
        """Collate function for DmpoDataset."""

        def collate_tokenized(items: list, key: str) -> dict[str, torch.Tensor]:
            return {
                'input_ids': torch.stack([x[key]['input_ids'] for x in items]),
                'attention_mask': torch.stack([x[key]['attention_mask'] for x in items]),
            }

        def collate_interaction(items: list) -> dict:
            result = {
                'query': collate_tokenized(items, 'query'),
                'accepted': collate_tokenized(items, 'accepted'),
                'rejected': collate_tokenized(items, 'rejected'),
            }
            if 'neutral' in items[0]:
                result['neutral'] = collate_tokenized(items, 'neutral')
            return result

        batch_interactions = [x['interactions'] for x in batch]
        max_interactions = max(len(inters) for inters in batch_interactions)

        if max_interactions > 0:
            # Pad shorter interaction lists
            padded_interactions = []
            for inters in batch_interactions:
                if len(inters) < max_interactions:
                    dummy = inters[0] if inters else batch[0]
                    padded = list(inters) + [dummy] * (max_interactions - len(inters))
                    padded_interactions.append(padded)
                else:
                    padded_interactions.append(inters)

            transposed = list(zip(*padded_interactions))

            return {
                **collate_interaction(batch),
                'interactions': [collate_interaction(list(step)) for step in transposed]
            }
        else:
            return {
                **collate_interaction(batch),
                'interactions': []
            }


class DmpoModel(nn.Module):
    """
    Model wrapper for DMPO training.

    Extends SupervisedMemoryAwareModel with support for:
    - Reference model (frozen copy for DPO loss)
    - Log probability computation for preference learning
    """

    def __init__(
            self,
            encoder: RxTEncoder,
            decoder: RxTDecoder,
            memory_attention: RxTMemoryAttentionType,
            train_only_decoder: bool = False,
            **kwargs
    ):
        super(DmpoModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.memory_attention = memory_attention
        self.train_only_decoder = train_only_decoder

        # Reference model will be set during training
        self.ref_decoder: Optional[RxTDecoder] = None

    def set_reference_model(self, ref_decoder: RxTDecoder):
        """Set frozen reference decoder for DPO loss computation."""
        self.ref_decoder = ref_decoder
        for param in self.ref_decoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        if self.train_only_decoder:
            return self.decoder.parameters()
        else:
            params_set = set(
                list(self.decoder.parameters()) +
                list(self.encoder.parameters()) +
                list(self.memory_attention.parameters())
            )
            return iter(params_set)

    def clone_reset_memory(self):
        self.memory_attention.clone_reset_memory()

    def reset_memory(self, init_type: str = None):
        self.memory_attention.reset_memory(init_type)

    def get_memory_state(self) -> torch.Tensor:
        """Get current memory state for cloning to reference model."""
        return self.memory_attention.model.stm.memory.clone()

    def set_memory_state(self, memory: torch.Tensor):
        """Set memory state (used for reference model alignment)."""
        self.memory_attention.model.stm.memory = memory

    def update_memory(self, x_e: torch.Tensor, encoder_mask: torch.Tensor = None):
        """Update memory based on encoded input (used for accepted interactions only)."""
        with torch.set_grad_enabled(not self.train_only_decoder):
            _, encoded_layers = self.encoder(x_e, attention_mask=encoder_mask)
            self.memory_attention(encoded_layers, attention_mask=encoder_mask)

    def forward_logits(
            self,
            x_d: torch.Tensor,
            decoder_mask: torch.Tensor = None,
            use_reference: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to get logits.

        Args:
            x_d: Decoder input tokens
            decoder_mask: Attention mask for decoder
            use_reference: If True, use frozen reference decoder

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        decoder = self.ref_decoder if use_reference else self.decoder
        return decoder(x_d, attention_mask=decoder_mask)

    def forward(
            self,
            x_e: torch.Tensor,
            x_d: torch.Tensor,
            encoder_mask: torch.Tensor = None,
            decoder_mask: torch.Tensor = None,
            is_first_step: bool = False
    ) -> torch.Tensor:
        """Standard forward pass (for compatibility with base trainer)."""
        if not is_first_step:
            self.update_memory(x_e, encoder_mask)
        return self.forward_logits(x_d, decoder_mask, use_reference=False)


class DmpoTrainer(BaseTrainer):
    """
    Direct Memory and Preference Optimization (DMPO) Trainer.

    Extends SMAT training with DPO-style preference learning. At each interaction step:
    1. Memory is updated based on the ACCEPTED interaction only
    2. Decoder generates log probabilities for both accepted and rejected responses
    3. Reference model (frozen) also generates log probabilities
    4. DPO loss is computed and backpropagated

    The DPO loss encourages the model to increase the probability gap between
    accepted and rejected responses while staying close to the reference distribution.

    Key Parameters:
    ---------------
    beta : float
        DPO temperature parameter. Higher values create stronger preference distinctions.
        Typical range: 0.1 - 0.5

    label_smoothing : float
        Smoothing parameter for the DPO loss. Helps prevent overconfidence.

    use_anchored : bool
        If True, use APO-style anchored loss with neutral responses.

    anchor_weight : float
        Weight for the anchor regularization term (only used if use_anchored=True).
    """

    def __init__(
            self,
            model: DmpoModel,
            device: torch.device,
            vocab_size: int,
            # DPO parameters
            beta: float = 0.1,
            label_smoothing: float = 0.0,
            # Anchored variant parameters
            use_anchored: bool = False,
            anchor_weight: float = 0.1,
            # Training parameters
            use_moe_aux_loss: bool = False,
            moe_aux_loss_scale: float = 0.01,
            max_seq_len: int = 256,
            pad_token_id: int = 0,
            train_only_decoder: bool = False,
            unfreeze_epochs: tuple[int, int] = (0, 0),
            dataset_collate_fn: Callable[[list[Any]], dict[str, Any]] = None,
            **kwargs
    ):
        if dataset_collate_fn is None:
            dataset_collate_fn = DmpoDataset.collate_dmpo_batch

        super(DmpoTrainer, self).__init__(
            model, device, dataset_collate_fn=dataset_collate_fn, **kwargs
        )
        self.vocab_size = vocab_size
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_anchored = use_anchored
        self.anchor_weight = anchor_weight
        self.use_moe_aux_loss = use_moe_aux_loss
        self.moe_aux_loss_scale = moe_aux_loss_scale
        self.get_batch_size = lambda batch: batch['query']['attention_mask'].size(0)
        self.total_inner_steps = 0
        self.valid_inner_steps = 0
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.train_only_decoder = train_only_decoder
        self.unfreeze_epochs = unfreeze_epochs

        # Reference model setup
        self._setup_reference_model()

        # Freeze components if needed
        if not self.train_only_decoder:
            mem_attn_unfreeze_epoch, encoder_unfreeze_epoch = self.unfreeze_epochs
            if mem_attn_unfreeze_epoch != 0:
                self._get_model().memory_attention.freeze()
            if encoder_unfreeze_epoch != 0:
                self._get_model().encoder.freeze_all()

    def _setup_reference_model(self):
        """Create frozen reference model for DPO loss."""
        model = self._get_model()
        # Deep copy the decoder for reference
        ref_decoder = copy.deepcopy(model.decoder)
        ref_decoder.eval()
        for param in ref_decoder.parameters():
            param.requires_grad = False
        model.set_reference_model(ref_decoder)

    def _get_model(self) -> DmpoModel:
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = next(model.children())
        return model

    def reset_stm(self):
        self._get_model().reset_memory()

    def _move_batch(self, batch: TokenizedDict) -> TokenizedDict:
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }

    def _move_multiple_batches(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._move_batch(batch) for batch in batches]

    def _compute_log_probs(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities from logits.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len] - target token ids
            attention_mask: [batch_size, seq_len]

        Returns:
            Per-sequence log probabilities [batch_size]
        """
        # Shift for autoregressive prediction
        shifted_logits = logits[:, :-1].contiguous()
        shifted_targets = targets[:, 1:].contiguous()
        shifted_mask = attention_mask[:, 1:].contiguous()

        # Compute per-token log probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shifted_targets.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding and sum per sequence
        masked_log_probs = token_log_probs * shifted_mask.float()
        seq_log_probs = masked_log_probs.sum(dim=-1)

        # Normalize by sequence length
        seq_lengths = shifted_mask.sum(dim=-1).clamp(min=1)
        normalized_log_probs = seq_log_probs / seq_lengths

        return normalized_log_probs

    def compute_dpo_loss(
            self,
            policy_accepted_logprobs: torch.Tensor,
            policy_rejected_logprobs: torch.Tensor,
            ref_accepted_logprobs: torch.Tensor,
            ref_rejected_logprobs: torch.Tensor,
            policy_neutral_logprobs: torch.Tensor = None,
            ref_neutral_logprobs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute DPO loss with optional anchoring.

        The standard DPO loss is:
        L = -log σ(β * ((log π(y_w) - log π(y_l)) - (log π_ref(y_w) - log π_ref(y_l))))

        With anchoring, we add:
        L_anchor = |log π(y_n) - log π_ref(y_n)|

        Args:
            policy_accepted_logprobs: Log probs from policy for accepted responses
            policy_rejected_logprobs: Log probs from policy for rejected responses
            ref_accepted_logprobs: Log probs from reference for accepted responses
            ref_rejected_logprobs: Log probs from reference for rejected responses
            policy_neutral_logprobs: Optional log probs from policy for neutral responses
            ref_neutral_logprobs: Optional log probs from reference for neutral responses

        Returns:
            DPO loss scalar
        """
        # Compute log-ratio differences
        policy_diff = policy_accepted_logprobs - policy_rejected_logprobs
        ref_diff = ref_accepted_logprobs - ref_rejected_logprobs

        # DPO loss
        logits = self.beta * (policy_diff - ref_diff)

        if self.label_smoothing > 0:
            # Apply label smoothing
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)

        dpo_loss = losses.mean()

        # Add anchor loss if enabled
        if self.use_anchored and policy_neutral_logprobs is not None:
            anchor_loss = torch.abs(
                policy_neutral_logprobs - ref_neutral_logprobs
            ).mean()
            dpo_loss = dpo_loss + self.anchor_weight * anchor_loss

        return dpo_loss

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        model = self._get_model()
        router_loss = model.decoder.model.moe_router_loss()
        loss = main_loss + self.moe_aux_loss_scale * router_loss

        if self.writer is not None:
            if self.model.training:
                if self.total_steps % self.tensorboard_interval == 0:
                    self.writer.add_scalar('Router aux loss/Train', router_loss.item(), self.total_steps)

        return loss

    def compute_loss(
            self,
            batch: dict,
            query_lens: torch.Tensor = None,
            is_first_step: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute DMPO loss for a batch.

        Args:
            batch: Dict with 'prev', 'query', 'accepted', 'rejected', and optionally 'neutral'
            query_lens: Lengths of query parts for masking
            is_first_step: Whether this is the first interaction step

        Returns:
            Tuple of (loss, metrics_dict)
        """
        model = self._get_model()

        # Get encoder input (previous interaction for memory update)
        encoder_inputs = batch['prev']['input_ids']
        encoder_mask = batch['prev']['attention_mask']

        # Get decoder inputs
        query = batch['query']
        accepted = batch['accepted']
        rejected = batch['rejected']

        # Build full decoder inputs (query + answer)
        accepted_input = smart_concat(query, accepted, max_length=self.max_seq_len, pad_token_id=self.pad_token_id)
        rejected_input = smart_concat(query, rejected, max_length=self.max_seq_len, pad_token_id=self.pad_token_id)

        # Clone memory state before update for reference model
        memory_before = model.get_memory_state()

        # Update memory with ACCEPTED interaction only (not rejected)
        if not is_first_step:
            accepted_full = smart_concat(query, accepted, max_length=self.max_seq_len, pad_token_id=self.pad_token_id)
            model.update_memory(accepted_full['input_ids'], accepted_full['attention_mask'])

        # Get memory state after update
        memory_after = model.get_memory_state()

        # Policy forward pass for accepted
        policy_accepted_logits = model.forward_logits(
            accepted_input['input_ids'],
            decoder_mask=accepted_input['attention_mask'],
            use_reference=False
        )

        # Policy forward pass for rejected
        policy_rejected_logits = model.forward_logits(
            rejected_input['input_ids'],
            decoder_mask=rejected_input['attention_mask'],
            use_reference=False
        )

        # Set reference model memory to same state
        model.ref_decoder.model.stm.memory = memory_after.detach()

        # Reference forward pass for accepted
        with torch.no_grad():
            ref_accepted_logits = model.forward_logits(
                accepted_input['input_ids'],
                decoder_mask=accepted_input['attention_mask'],
                use_reference=True
            )
            ref_rejected_logits = model.forward_logits(
                rejected_input['input_ids'],
                decoder_mask=rejected_input['attention_mask'],
                use_reference=True
            )

        # Compute log probabilities (mask query tokens)
        def mask_query_tokens(targets: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
            masked = targets.clone()
            for i in range(masked.size(0)):
                masked[i, :q_lens[i].item()] = -100
            return masked

        accepted_targets = mask_query_tokens(accepted_input['input_ids'], query_lens)
        rejected_targets = mask_query_tokens(rejected_input['input_ids'], query_lens)

        # Create masks that exclude query tokens
        accepted_answer_mask = accepted_input['attention_mask'].clone()
        rejected_answer_mask = rejected_input['attention_mask'].clone()
        for i in range(accepted_answer_mask.size(0)):
            accepted_answer_mask[i, :query_lens[i].item()] = 0
            rejected_answer_mask[i, :query_lens[i].item()] = 0

        policy_accepted_logprobs = self._compute_log_probs(
            policy_accepted_logits, accepted_input['input_ids'], accepted_answer_mask
        )
        policy_rejected_logprobs = self._compute_log_probs(
            policy_rejected_logits, rejected_input['input_ids'], rejected_answer_mask
        )
        ref_accepted_logprobs = self._compute_log_probs(
            ref_accepted_logits, accepted_input['input_ids'], accepted_answer_mask
        )
        ref_rejected_logprobs = self._compute_log_probs(
            ref_rejected_logits, rejected_input['input_ids'], rejected_answer_mask
        )

        # Handle anchored variant
        policy_neutral_logprobs = None
        ref_neutral_logprobs = None

        if self.use_anchored and 'neutral' in batch:
            neutral = batch['neutral']
            neutral_input = smart_concat(query, neutral, max_length=self.max_seq_len, pad_token_id=self.pad_token_id)

            policy_neutral_logits = model.forward_logits(
                neutral_input['input_ids'],
                decoder_mask=neutral_input['attention_mask'],
                use_reference=False
            )

            with torch.no_grad():
                ref_neutral_logits = model.forward_logits(
                    neutral_input['input_ids'],
                    decoder_mask=neutral_input['attention_mask'],
                    use_reference=True
                )

            neutral_answer_mask = neutral_input['attention_mask'].clone()
            for i in range(neutral_answer_mask.size(0)):
                neutral_answer_mask[i, :query_lens[i].item()] = 0

            policy_neutral_logprobs = self._compute_log_probs(
                policy_neutral_logits, neutral_input['input_ids'], neutral_answer_mask
            )
            ref_neutral_logprobs = self._compute_log_probs(
                ref_neutral_logits, neutral_input['input_ids'], neutral_answer_mask
            )

        # Compute DPO loss
        dpo_loss = self.compute_dpo_loss(
            policy_accepted_logprobs,
            policy_rejected_logprobs,
            ref_accepted_logprobs,
            ref_rejected_logprobs,
            policy_neutral_logprobs,
            ref_neutral_logprobs,
        )

        # Apply MoE auxiliary loss if enabled
        total_loss = self._moe_aux_loss(dpo_loss)

        # Compute metrics
        with torch.no_grad():
            reward_margin = (policy_accepted_logprobs - policy_rejected_logprobs).mean().item()
            accepted_reward = policy_accepted_logprobs.mean().item()
            rejected_reward = policy_rejected_logprobs.mean().item()

        metrics = {
            'dpo_loss': dpo_loss.item(),
            'reward_margin': reward_margin,
            'accepted_reward': accepted_reward,
            'rejected_reward': rejected_reward,
        }

        return total_loss, metrics

    def _run_epoch(
            self,
            dataloader: torch.utils.data.DataLoader,
            epoch: int,
            optimizer: torch.optim.Optimizer,
            batch_size: int,
            scaler: torch.cuda.amp.GradScaler = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(self.model, epoch)

        # Handle unfreezing
        if not self.train_only_decoder:
            mem_attn_unfreeze_epoch, encoder_unfreeze_epoch = self.unfreeze_epochs
            if mem_attn_unfreeze_epoch == epoch:
                self._get_model().memory_attention.unfreeze()
            if encoder_unfreeze_epoch == epoch:
                self._get_model().encoder.unfreeze_all(True, True)

        self.accumulated_loss = torch.tensor(0.0, device=self.device)
        self.optimizer_step_count = 0
        accumulated_tokens = torch.tensor(0, dtype=torch.long, device=self.device)

        for batch_idx, batch in enumerate(dataloader):
            if not self.is_running:
                break

            if self.get_batch_size(batch) == batch_size:
                self.total_steps += 1
                self.epoch_steps = batch_idx

                self.reset_stm()

                # Get first interaction and follow-ups
                first_query = self._move_batch(batch['query'])
                first_accepted = self._move_batch(batch['accepted'])
                first_rejected = self._move_batch(batch['rejected'])
                first_neutral = self._move_batch(batch['neutral']) if 'neutral' in batch else None
                interactions = batch['interactions']

                number_of_inner_steps = len(interactions) + 1

                prev_query, prev_accepted = first_query, first_accepted

                for inner_step_idx in range(number_of_inner_steps):
                    self.total_inner_steps += 1

                    if inner_step_idx == 0:
                        curr_query = first_query
                        curr_accepted = first_accepted
                        curr_rejected = first_rejected
                        curr_neutral = first_neutral
                    else:
                        inter = interactions[inner_step_idx - 1]
                        curr_query = self._move_batch(inter['query'])
                        curr_accepted = self._move_batch(inter['accepted'])
                        curr_rejected = self._move_batch(inter['rejected'])
                        curr_neutral = self._move_batch(inter['neutral']) if 'neutral' in inter else None

                    self._get_model().clone_reset_memory()

                    query_lens = curr_query['attention_mask'].sum(dim=-1)

                    # Build training batch
                    train_batch = {
                        'prev': smart_concat(prev_query, prev_accepted, max_length=self.max_seq_len, pad_token_id=self.pad_token_id),
                        'query': curr_query,
                        'accepted': curr_accepted,
                        'rejected': curr_rejected,
                    }
                    if curr_neutral is not None:
                        train_batch['neutral'] = curr_neutral

                    accumulated_tokens += curr_query['attention_mask'].sum()
                    accumulated_tokens += curr_accepted['attention_mask'].sum()

                    # Compute loss
                    if self.use_amp:
                        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                            loss, metrics = self.compute_loss(
                                train_batch,
                                query_lens=query_lens,
                                is_first_step=inner_step_idx == 0
                            )
                    else:
                        loss, metrics = self.compute_loss(
                            train_batch,
                            query_lens=query_lens,
                            is_first_step=inner_step_idx == 0
                        )

                    self.accumulated_loss += loss
                    loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp and scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    self.optimizer_step_count += 1
                    if self.optimizer_step_count % self.gradient_accumulation_steps == 0:
                        if self.use_amp and scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._get_model().trainable_parameters(),
                            max_norm=1.0,
                            error_if_nonfinite=False
                        )
                        if self.use_amp and scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        optimizer.zero_grad()

                        if scheduler is not None:
                            scheduler.step()

                        if self.writer and self.total_inner_steps % self.tensorboard_interval == 0:
                            loss_item = (self.accumulated_loss / self.gradient_accumulation_steps).item()
                            self._train_writer(loss_item, metrics, inner_step_idx)
                            self.total_tokens += accumulated_tokens.item()
                            accumulated_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
                            self.writer.add_scalar('Processed tokens', self.total_tokens, self.total_inner_steps)

                        self.accumulated_loss = torch.tensor(0.0, device=self.device)
                        self.optimizer_step_count = 0

                    # Update previous interaction for next step
                    prev_query, prev_accepted = curr_query, curr_accepted

                    for callback in self.callbacks:
                        should_stop = callback.on_batch_end(
                            self.model,
                            (batch_idx * number_of_inner_steps) + inner_step_idx,
                            loss,
                            train_batch
                        )
                        if should_stop:
                            self.is_running = False

        # Validation
        if self.validation_dataset:
            self.validation_steps = 0
            self.valid_inner_steps = 0
            val_loss, val_metrics = self.validate(batch_size)
            if self.use_ddp:
                val_loss = distributed_value_mean(val_loss, device=self.device)

            self.validation_metrics[epoch] = val_metrics

            if self.writer:
                self._valid_writer(epoch, val_loss, val_metrics)

            for callback in self.callbacks:
                should_stop = callback.on_validation_end(self.model, epoch, val_loss, val_metrics)
                if should_stop:
                    self.is_running = False

        for callback in self.callbacks:
            should_stop = callback.on_epoch_end(self.model, epoch)
            if should_stop:
                self.is_running = False

        if self.writer:
            self.writer.flush()

    def _train_writer(self, loss: float, metrics: dict, inner_step: int) -> None:
        self.writer.add_scalar('Loss/train', loss, self.total_inner_steps)
        self.writer.add_scalar('DPO Loss/train', metrics['dpo_loss'], self.total_inner_steps)
        self.writer.add_scalar('Reward Margin/train', metrics['reward_margin'], self.total_inner_steps)
        self.writer.add_scalar('Accepted Reward/train', metrics['accepted_reward'], self.total_inner_steps)
        self.writer.add_scalar('Rejected Reward/train', metrics['rejected_reward'], self.total_inner_steps)
        self.writer.add_scalar(f'Loss/train (step {inner_step})', loss, self.total_inner_steps)

    def _valid_writer(self, epoch: int, val_loss: float, val_metrics: dict):
        self.writer.add_scalar('Loss/Valid', val_loss, epoch)
        if 'reward_margin' in val_metrics:
            self.writer.add_scalar('Reward Margin/Valid', val_metrics['reward_margin'], epoch)
        if 'accuracy' in val_metrics:
            self.writer.add_scalar('Preference Accuracy/Valid', val_metrics['accuracy'], epoch)

    def validate(self, batch_size: int) -> tuple[float, dict]:
        self.model.eval()
        all_val_loss = torch.tensor(0.0, device=self.device)
        all_reward_margin = torch.tensor(0.0, device=self.device)
        correct_preferences = torch.tensor(0, device=self.device)
        total_comparisons = torch.tensor(0, device=self.device)

        val_dataloader = self._valid_loader(batch_size)
        processed = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if self.get_batch_size(batch) == batch_size:
                    processed += 1
                    val_loss = torch.tensor(0.0, device=self.device)
                    val_margin = torch.tensor(0.0, device=self.device)

                    self.reset_stm()

                    first_query = self._move_batch(batch['query'])
                    first_accepted = self._move_batch(batch['accepted'])
                    first_rejected = self._move_batch(batch['rejected'])
                    first_neutral = self._move_batch(batch['neutral']) if 'neutral' in batch else None
                    interactions = batch['interactions']

                    number_of_inner_steps = len(interactions) + 1
                    prev_query, prev_accepted = first_query, first_accepted

                    for inner_step_idx in range(number_of_inner_steps):
                        self.valid_inner_steps += 1

                        if inner_step_idx == 0:
                            curr_query = first_query
                            curr_accepted = first_accepted
                            curr_rejected = first_rejected
                            curr_neutral = first_neutral
                        else:
                            inter = interactions[inner_step_idx - 1]
                            curr_query = self._move_batch(inter['query'])
                            curr_accepted = self._move_batch(inter['accepted'])
                            curr_rejected = self._move_batch(inter['rejected'])
                            curr_neutral = self._move_batch(inter['neutral']) if 'neutral' in inter else None

                        self._get_model().clone_reset_memory()

                        query_lens = curr_query['attention_mask'].sum(dim=-1)

                        valid_batch = {
                            'prev': smart_concat(prev_query, prev_accepted, max_length=self.max_seq_len, pad_token_id=self.pad_token_id),
                            'query': curr_query,
                            'accepted': curr_accepted,
                            'rejected': curr_rejected,
                        }
                        if curr_neutral is not None:
                            valid_batch['neutral'] = curr_neutral

                        loss, metrics = self.compute_loss(
                            valid_batch,
                            query_lens=query_lens,
                            is_first_step=inner_step_idx == 0
                        )

                        val_loss += loss
                        val_margin += metrics['reward_margin']

                        # Track preference accuracy
                        if metrics['accepted_reward'] > metrics['rejected_reward']:
                            correct_preferences += batch_size
                        total_comparisons += batch_size

                        prev_query, prev_accepted = curr_query, curr_accepted

                    all_val_loss += val_loss / number_of_inner_steps
                    all_reward_margin += val_margin / number_of_inner_steps

        avg_loss = all_val_loss / max(processed, 1)
        avg_margin = all_reward_margin / max(processed, 1)
        accuracy = (correct_preferences / max(total_comparisons, 1) * 100).item()

        if self.use_ddp:
            avg_loss = distributed_mean(avg_loss)
            avg_margin = distributed_mean(avg_margin)

        metrics = {
            'loss': avg_loss.item(),
            'reward_margin': avg_margin.item(),
            'accuracy': accuracy,
        }

        self.model.train()
        return avg_loss.item(), metrics


class AnchoredDmpoTrainer(DmpoTrainer):
    """
    Anchored DMPO Trainer - extends DMPO with APO-style neutral anchoring.

    This variant adds a neutral reference response that acts as an anchor point,
    preventing the model from drifting too far from reasonable behavior while
    still learning strong preferences.

    The neutral response serves as:
    1. A regularization point to prevent reward hacking
    2. A baseline for measuring preference strength
    3. A safety mechanism to maintain response quality

    Usage:
    ------
    Simply use this trainer instead of DmpoTrainer and provide datasets with
    'neutral' field in interactions. The neutral response should represent
    a reasonable but not exceptional response to the query.
    """

    def __init__(self, *args, **kwargs):
        # Force anchored mode
        kwargs['use_anchored'] = True
        super(AnchoredDmpoTrainer, self).__init__(*args, **kwargs)
