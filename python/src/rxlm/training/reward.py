import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Optional, Literal, Union
from .utils import TokenizedDict
from ..metrics.tensorbleu import tensor_sentence_bleu

# Optional NLTK fallback for debugging
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sentence_bleu = None
    SmoothingFunction = None


class MrlRewardMode(Enum):
    STANDARD = 1
    NEGATIVE = 2
    LONG_RANGE = 3


class BleuBackend(Enum):
    """Backend selection for BLEU computation."""
    TENSOR = 1  # Fast tensor-based BLEU (default, GPU-accelerated)
    NLTK = 2    # NLTK sentence_bleu (slower, for debugging/comparison)


class PreferenceRewardMode(Enum):
    """Mode for combining memory and preference rewards."""
    MEMORY_ONLY = 1      # Only use memory retention rewards (default)
    PREFERENCE_ONLY = 2  # Only use preference rewards
    COMBINED = 3         # Combine both with configurable weights


class MrlRewardModel:
    """
    Memory Reinforcement Learning Reward Model.

    Computes rewards based on BLEU and cosine similarity between generated responses,
    reference answers, and saved memory data.

    Supports two backends for BLEU computation:
    - TENSOR (default): Fast, GPU-accelerated tensor-based BLEU from TensorBLEU paper
    - NLTK: Traditional NLTK sentence_bleu for debugging/comparison

    Args:
        shared_embedding: Shared embedding layer for cosine similarity
        bleu_backend: Backend for BLEU computation (BleuBackend.TENSOR or BleuBackend.NLTK)
        pad_token_id: Padding token ID for tensor BLEU masking
        bleu_max_n: Maximum n-gram order for BLEU (default: 2 for bigrams)
        bleu_smoothing: Smoothing method for tensor BLEU ('exp', 'floor', 'add-k', 'none')
        ... (other args same as before)
    """

    def __init__(
            self,
            shared_embedding: nn.Embedding,
            bleu_with_saved_data: bool = False,
            bleu_mode: Literal['separate', 'combined'] = 'separate',
            bleu_factor: float = 0.5,
            bleu_ref_factor: float = 0.5,
            bleu_saved_factor: float = 0.5,
            bleu_first_ref_factor: Optional[float] = None,
            bleu_first_saved_factor: Optional[float] = None,
            cos_factor: float = 0.5,
            cos_ref_factor: float = 0.5,
            cos_saved_factor: float = 0.5,
            multi_cos_ref_factor: float = 0.3,
            multi_cos_saved_factor: float = 0.5,
            multi_cos_running_mean_factor: float = 0.2,
            neg_bleu_factor: Optional[float] = None,
            neg_cos_factor: Optional[float] = None,
            neg_cos_ref_factor: Optional[float] = None,
            neg_cos_saved_factor: Optional[float] = None,
            neg_bleu_ref_factor: float = 0.5,
            neg_bleu_saved_factor: float = 0.5,
            allow_not_summing_factors: bool = False,
            reward_len: bool = False,
            neg_reward_len: bool = False,
            max_rewarded_len: int = None,
            target_len_as_ref: bool = False,
            len_factor: int = None,
            use_running_mean: bool = True,
            running_mean_decay: float = 0.2,
            bleu_saved_weights: tuple = (0.5, 0.5),
            bleu_ref_weights: tuple = (0.5, 0.5),
            tanh_reward_scale: bool = False,
            rewards_scale: float = 1.0,
            debug_mode: int = 0,
            # New TensorBLEU parameters
            bleu_backend: BleuBackend = BleuBackend.TENSOR,
            pad_token_id: int = 0,
            bleu_max_n: int = 2,
            bleu_smoothing: Literal['exp', 'floor', 'add-k', 'none'] = 'exp',
    ):
        self.shared_embedding = shared_embedding
        self.bleu_with_saved_data = bleu_with_saved_data
        self.bleu_mode = bleu_mode

        self.bleu_factor = bleu_factor
        self.bleu_ref_factor = bleu_ref_factor
        self.bleu_saved_factor = bleu_saved_factor
        self.bleu_first_ref_factor = bleu_first_ref_factor if bleu_first_ref_factor is not None else bleu_ref_factor
        self.bleu_first_saved_factor = bleu_first_saved_factor if bleu_first_saved_factor is not None else bleu_saved_factor
        self.cos_factor = cos_factor
        self.cos_ref_factor = cos_ref_factor
        self.cos_saved_factor = cos_saved_factor
        self.multi_cos_ref_factor = multi_cos_ref_factor
        self.multi_cos_saved_factor = multi_cos_saved_factor
        self.multi_cos_running_mean_factor = multi_cos_running_mean_factor
        self.neg_bleu_factor = neg_bleu_factor if neg_bleu_factor is not None else bleu_factor
        self.neg_cos_factor = neg_cos_factor if neg_cos_factor is not None else cos_factor
        self.neg_cos_ref_factor = neg_cos_ref_factor if neg_cos_ref_factor is not None else cos_ref_factor
        self.neg_cos_saved_factor = neg_cos_saved_factor if neg_cos_saved_factor is not None else cos_saved_factor
        self.neg_bleu_ref_factor = neg_bleu_ref_factor
        self.neg_bleu_saved_factor = neg_bleu_saved_factor
        self.reward_len = reward_len
        self.neg_reward_len = neg_reward_len
        self.max_rewarded_len = max_rewarded_len
        self.target_len_as_ref = target_len_as_ref
        self.len_factor = len_factor
        self.use_running_mean = use_running_mean
        self.running_mean_decay = running_mean_decay
        self.bleu_ref_weights = bleu_ref_weights
        self.bleu_saved_weights = bleu_saved_weights
        self.tanh_reward_scale = tanh_reward_scale
        self.rewards_scale = rewards_scale
        self.debug_mode = debug_mode

        # TensorBLEU configuration
        self.bleu_backend = bleu_backend
        self.pad_token_id = pad_token_id
        self.bleu_max_n = bleu_max_n
        self.bleu_smoothing = bleu_smoothing

        # Convert tuple weights to tensor for TensorBLEU
        self._tensor_ref_weights = None
        self._tensor_saved_weights = None

        # NLTK smoothing (only initialize if NLTK available and backend is NLTK)
        if bleu_backend == BleuBackend.NLTK:
            if not NLTK_AVAILABLE:
                raise ImportError("NLTK not available. Install nltk or use BleuBackend.TENSOR")
            self.bleu_smoothing_fn = SmoothingFunction().method4
        else:
            self.bleu_smoothing_fn = None

        self.prev_data_running_mean = None

        if not allow_not_summing_factors:
            if reward_len:
                assert self.bleu_factor + self.cos_factor + self.len_factor == 1.0
                assert self.neg_bleu_factor + self.neg_cos_factor + self.len_factor == 1.0
                assert self.multi_cos_ref_factor + self.multi_cos_saved_factor + self.multi_cos_running_mean_factor == 1.0
                assert self.bleu_ref_factor + self.bleu_saved_factor == 1.0
                assert self.cos_ref_factor + self.cos_saved_factor == 1.0
                assert self.neg_cos_ref_factor + self.neg_cos_saved_factor == 1.0
                assert self.neg_bleu_ref_factor + self.neg_bleu_saved_factor == 1.0
                assert self.bleu_first_ref_factor + self.bleu_first_saved_factor == 1.0
            else:
                assert self.bleu_factor + self.cos_factor == 1.0
                assert self.bleu_ref_factor + self.bleu_saved_factor == 1.0
                assert self.cos_ref_factor + self.cos_saved_factor == 1.0
                assert self.multi_cos_ref_factor + self.multi_cos_saved_factor + self.multi_cos_running_mean_factor == 1.0
                assert self.neg_bleu_factor + self.neg_cos_factor == 1.0
                assert self.neg_cos_ref_factor + self.neg_cos_saved_factor == 1.0
                assert self.neg_bleu_ref_factor + self.neg_bleu_saved_factor == 1.0
                assert self.bleu_first_ref_factor + self.bleu_first_saved_factor == 1.0

    def _get_tensor_weights(self, weights: tuple, device: torch.device) -> torch.Tensor:
        """Convert tuple weights to tensor, caching for efficiency."""
        # Pad weights to match bleu_max_n if needed
        if len(weights) < self.bleu_max_n:
            weights = tuple(weights) + (0.0,) * (self.bleu_max_n - len(weights))
        elif len(weights) > self.bleu_max_n:
            weights = weights[:self.bleu_max_n]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def _tensor_sentence_bleu_batch(
            self,
            candidates: torch.Tensor,
            references: torch.Tensor,
            weights: tuple,
    ) -> torch.Tensor:
        """
        Compute BLEU scores using TensorBLEU for entire batch at once.

        Args:
            candidates: Generated token IDs [batch_size, seq_len]
            references: Reference token IDs [batch_size, seq_len] or [batch_size, num_refs, seq_len]

        Returns:
            BLEU scores tensor [batch_size]
        """
        device = candidates.device

        # Ensure references have the right shape [batch_size, num_refs, ref_len]
        if references.dim() == 2:
            references = references.unsqueeze(1)  # Add num_refs dimension

        tensor_weights = self._get_tensor_weights(weights, device)

        return tensor_sentence_bleu(
            candidates=candidates,
            references=references,
            pad_token_id=self.pad_token_id,
            max_n=self.bleu_max_n,
            weights=tensor_weights,
            smoothing_method=self.bleu_smoothing,
        )

    def _sentence_bleu(self, input_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                       masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], is_first_step: bool = False) -> float:
        generated, reference, saved_data = input_ids
        generated_mask, reference_mask, saved_data_mask = masks

        generated = generated.tolist()[:generated_mask.sum().item()]
        reference = reference.tolist()[:reference_mask.sum().item()]
        saved_data = saved_data.tolist()[:saved_data_mask.sum().item()]

        if self.debug_mode == 2:
            print('LENS: ', (len(generated), len(reference), len(saved_data)))

        if self.bleu_with_saved_data:
            if self.bleu_mode == 'separate':
                ref_bleu = sentence_bleu([reference], generated, weights=self.bleu_ref_weights,
                                         smoothing_function=self.bleu_smoothing)
                saved_bleu = sentence_bleu([saved_data], generated, weights=self.bleu_saved_weights,
                                           smoothing_function=self.bleu_smoothing)
                if self.debug_mode == 2:
                    print('REF BLEU: ', ref_bleu)
                    print('SAVED BLEU: ', saved_bleu)

                if is_first_step:
                    return self.bleu_first_ref_factor * ref_bleu + self.bleu_first_saved_factor * saved_bleu
                else:
                    return self.bleu_ref_factor * ref_bleu + self.bleu_saved_factor * saved_bleu
            else:
                return sentence_bleu(
                    [reference, saved_data], generated,
                    weights=self.bleu_ref_weights, smoothing_function=self.bleu_smoothing
                )
        else:
            return sentence_bleu(
                [reference], generated,
                weights=self.bleu_ref_weights, smoothing_function=self.bleu_smoothing
            )

    def _negative_sentence_bleu(self, input_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        generated, reference, saved_data = input_ids
        generated_mask, reference_mask, saved_data_mask = masks

        generated = generated.tolist()[:generated_mask.sum().item()]
        reference = reference.tolist()[:reference_mask.sum().item()]
        saved_data = saved_data.tolist()[:saved_data_mask.sum().item()]

        if self.debug_mode == 2:
            print('LENS: ', (len(generated), len(reference), len(saved_data)))

        if self.bleu_with_saved_data:
            ref_bleu = sentence_bleu([reference], generated, weights=self.bleu_ref_weights,
                                     smoothing_function=self.bleu_smoothing)
            saved_bleu = sentence_bleu([saved_data], generated, weights=self.bleu_saved_weights,
                                       smoothing_function=self.bleu_smoothing)
            saved_bleu = 1 - saved_bleu

            if self.debug_mode == 2:
                print('REF BLEU: ', ref_bleu)
                print('SAVED BLEU: ', saved_bleu)

            return self.neg_bleu_ref_factor * ref_bleu + self.neg_bleu_saved_factor * saved_bleu
        else:
            return sentence_bleu([reference], generated, weights=self.bleu_ref_weights)

    def batch_bleu(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict,
                   is_first_step: bool = False) -> Union[list[float], torch.Tensor]:
        """
        Compute BLEU scores for a batch.

        Uses TensorBLEU (GPU-accelerated) by default, or NLTK for debugging.

        Returns:
            If TensorBLEU: torch.Tensor [batch_size]
            If NLTK: list[float]
        """
        if self.bleu_backend == BleuBackend.TENSOR:
            return self._batch_bleu_tensor(generated, reference, saved_data, is_first_step)
        else:
            return self._batch_bleu_nltk(generated, reference, saved_data, is_first_step)

    def _batch_bleu_tensor(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict,
                           is_first_step: bool = False) -> torch.Tensor:
        """Compute batch BLEU using TensorBLEU - fully vectorized and GPU-accelerated."""
        gen_ids = generated['input_ids']
        ref_ids = reference['input_ids']
        saved_ids = saved_data['input_ids']

        if self.bleu_with_saved_data:
            if self.bleu_mode == 'separate':
                # Compute BLEU with reference and saved data separately
                ref_bleu = self._tensor_sentence_bleu_batch(gen_ids, ref_ids, self.bleu_ref_weights)
                saved_bleu = self._tensor_sentence_bleu_batch(gen_ids, saved_ids, self.bleu_saved_weights)

                if self.debug_mode == 2:
                    print(f'[TensorBLEU] REF BLEU: {ref_bleu.mean().item():.4f}')
                    print(f'[TensorBLEU] SAVED BLEU: {saved_bleu.mean().item():.4f}')

                if is_first_step:
                    return self.bleu_first_ref_factor * ref_bleu + self.bleu_first_saved_factor * saved_bleu
                else:
                    return self.bleu_ref_factor * ref_bleu + self.bleu_saved_factor * saved_bleu
            else:
                # Combined mode: stack references
                combined_refs = torch.stack([ref_ids, saved_ids], dim=1)  # [B, 2, seq_len]
                return self._tensor_sentence_bleu_batch(gen_ids, combined_refs, self.bleu_ref_weights)
        else:
            return self._tensor_sentence_bleu_batch(gen_ids, ref_ids, self.bleu_ref_weights)

    def _batch_bleu_nltk(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict,
                         is_first_step: bool = False) -> list[float]:
        """Compute batch BLEU using NLTK - for debugging/comparison."""
        batch_size = generated['input_ids'].size(0)

        return [
            self._sentence_bleu(
                input_ids=(generated['input_ids'][i], reference['input_ids'][i], saved_data['input_ids'][i]),
                masks=(generated['attention_mask'][i], reference['attention_mask'][i], saved_data['attention_mask'][i]),
                is_first_step=is_first_step,
            ) for i in range(batch_size)
        ]

    def negative_bleu(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict) -> Union[
        list[float], torch.Tensor]:
        """
        Compute negative BLEU scores for a batch (for long-range retention mode).
        """
        if self.bleu_backend == BleuBackend.TENSOR:
            return self._negative_bleu_tensor(generated, reference, saved_data)
        else:
            return self._negative_bleu_nltk(generated, reference, saved_data)

    def _negative_bleu_tensor(self, generated: TokenizedDict, reference: TokenizedDict,
                              saved_data: TokenizedDict) -> torch.Tensor:
        """Compute negative BLEU using TensorBLEU."""
        gen_ids = generated['input_ids']
        ref_ids = reference['input_ids']
        saved_ids = saved_data['input_ids']

        if self.bleu_with_saved_data:
            ref_bleu = self._tensor_sentence_bleu_batch(gen_ids, ref_ids, self.bleu_ref_weights)
            saved_bleu = self._tensor_sentence_bleu_batch(gen_ids, saved_ids, self.bleu_saved_weights)
            # Invert saved_bleu for negative mode
            saved_bleu = 1 - saved_bleu

            if self.debug_mode == 2:
                print(f'[TensorBLEU Negative] REF BLEU: {ref_bleu.mean().item():.4f}')
                print(f'[TensorBLEU Negative] SAVED BLEU (inverted): {saved_bleu.mean().item():.4f}')

            return self.neg_bleu_ref_factor * ref_bleu + self.neg_bleu_saved_factor * saved_bleu
        else:
            return self._tensor_sentence_bleu_batch(gen_ids, ref_ids, self.bleu_ref_weights)

    def _negative_bleu_nltk(self, generated: TokenizedDict, reference: TokenizedDict,
                            saved_data: TokenizedDict) -> list[float]:
        """Compute negative BLEU using NLTK."""
        batch_size = generated['input_ids'].size(0)

        return [
            self._negative_sentence_bleu(
                input_ids=(generated['input_ids'][i], reference['input_ids'][i], saved_data['input_ids'][i]),
                masks=(generated['attention_mask'][i], reference['attention_mask'][i], saved_data['attention_mask'][i])
            ) for i in range(batch_size)
        ]

    def _sequence_embedding(self, sequence: TokenizedDict) -> torch.Tensor:
        input_ids = sequence['input_ids']
        attention_mask = sequence['attention_mask']

        # Get embeddings
        embeddings = self.shared_embedding(input_ids)

        # Apply attention mask
        mask_expanded = attention_mask.unsqueeze(-1)
        masked_embeddings = embeddings * mask_expanded

        # Compute mean with masking
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_counts = torch.sum(mask_expanded, dim=1)
        token_counts = torch.clamp(token_counts, min=1e-8)  # Avoid division by zero

        return sum_embeddings / token_counts

    def _cosine_sim(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict):
        generated_emb = F.normalize(self._sequence_embedding(generated), dim=-1)
        saved_data_emb = F.normalize(self._sequence_embedding(saved_data), dim=-1)
        reference_emb = F.normalize(self._sequence_embedding(reference), dim=-1)

        gen_and_saved = F.cosine_similarity(generated_emb, saved_data_emb, dim=1)
        gen_and_ref = F.cosine_similarity(generated_emb, reference_emb, dim=1)

        if self.debug_mode >= 1:
            print('GEN AND SAVED: ', gen_and_saved.mean())
            print('GEN AND REF: ', gen_and_ref.mean())
        return torch.clamp(gen_and_saved, min=0), torch.clamp(gen_and_ref, min=0)

    def _cosine_sim_running_mean(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict):
        generated_emb = F.normalize(self._sequence_embedding(generated), dim=-1)
        saved_data_emb = F.normalize(self._sequence_embedding(saved_data), dim=-1)
        reference_emb = F.normalize(self._sequence_embedding(reference), dim=-1)
        running_emb = F.normalize(self.prev_data_running_mean, dim=-1)

        gen_and_saved = F.cosine_similarity(generated_emb, saved_data_emb, dim=1)
        gen_and_ref = F.cosine_similarity(generated_emb, reference_emb, dim=1)
        gen_and_mean = F.cosine_similarity(generated_emb, running_emb, dim=1)

        if self.debug_mode >= 1:
            print('GEN AND SAVED: ', gen_and_saved.mean())
            print('GEN AND REF: ', gen_and_ref.mean())
            print('GEN AND MEAN: ', gen_and_mean.mean())

        return torch.clamp(gen_and_saved, min=0), torch.clamp(gen_and_ref, min=0), torch.clamp(gen_and_mean, min=0)

    def batch_cosine(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict,
                     include_running_mean: bool = False, negative_running_mean: bool = False) -> torch.Tensor:
        if self.use_running_mean and negative_running_mean:
            gen_and_saved, gen_and_ref, gen_and_mean = self._cosine_sim_running_mean(generated, reference, saved_data)
            return self.multi_cos_saved_factor * gen_and_saved + self.multi_cos_ref_factor * gen_and_ref + self.multi_cos_running_mean_factor * (
                    1 - gen_and_mean)
        elif self.use_running_mean and include_running_mean:
            gen_and_saved, gen_and_ref, gen_and_mean = self._cosine_sim_running_mean(generated, reference, saved_data)
            return self.multi_cos_saved_factor * gen_and_saved + self.multi_cos_ref_factor * gen_and_ref + self.multi_cos_running_mean_factor * gen_and_mean
        else:
            gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)
            return self.cos_saved_factor * gen_and_saved + self.cos_ref_factor * gen_and_ref

    def negative_cosine(self, generated: TokenizedDict, reference: TokenizedDict,
                        saved_data: TokenizedDict) -> torch.Tensor:
        gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)

        return self.neg_cos_saved_factor * (1 - gen_and_saved) + self.neg_cos_ref_factor * gen_and_ref

    def len_reward(self, generated: TokenizedDict, reference: TokenizedDict) -> torch.Tensor:
        target_lens = reference['attention_mask'].sum(dim=-1) if self.target_len_as_ref else self.max_rewarded_len
        lens = generated['attention_mask'].sum(dim=-1)
        neg_lens = target_lens / lens if self.neg_reward_len else 1.0
        len_reward = torch.where(lens >= target_lens, neg_lens, lens / target_lens)
        return len_reward

    def reset_running_mean(self):
        self.prev_data_running_mean = None

    def init_running_mean(self, prev_data: TokenizedDict):
        self.prev_data_running_mean = self._sequence_embedding(prev_data)

    def update_running_mean(self, prev_data: TokenizedDict):
        self.prev_data_running_mean = (1 - self.running_mean_decay) * self._sequence_embedding(
            prev_data) + self.running_mean_decay * self.prev_data_running_mean

    def _pre_scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.tanh_reward_scale:
            return (rewards * 2) - 1  # Convert [0,1] to [-1,1]
        else:
            return rewards

    def _to_tensor(self, value: Union[list[float], torch.Tensor], device: torch.device) -> torch.Tensor:
        """Convert list or tensor to tensor on specified device."""
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.tensor(value, device=device, dtype=torch.float32)

    def __call__(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
            saved_data: TokenizedDict,
            prev_data: TokenizedDict = None,
            mode: MrlRewardMode = MrlRewardMode.STANDARD
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.

        Args:
            generated: Generated token sequences
            reference: Reference answer sequences
            saved_data: Saved memory data sequences
            prev_data: Previous interaction data (for running mean)
            mode: Reward mode (STANDARD, NEGATIVE, LONG_RANGE)

        Returns:
            Reward tensor [batch_size]
        """
        if prev_data is not None:
            if self.prev_data_running_mean is None:
                self.init_running_mean(prev_data)
            else:
                self.update_running_mean(prev_data)

        device = generated['input_ids'].device

        if mode == MrlRewardMode.STANDARD:
            bleu = self.batch_bleu(generated, reference, saved_data, is_first_step=prev_data is None)
            cosine = self.batch_cosine(generated, reference, saved_data, include_running_mean=prev_data is not None)

            # Convert to tensor if needed (handles both NLTK list and TensorBLEU tensor)
            bleu_tensor = self._to_tensor(bleu, device)

            if self.debug_mode >= 1:
                print('--- STANDARD MODE')
                print(f'--- BLEU:  {bleu_tensor.mean().item():.4f}  / max: {bleu_tensor.max().item():.4f} / min: {bleu_tensor.min().item():.4f}')
                print(f'--- COSINE: {cosine.mean().item():.4f} / max: {cosine.max().item():.4f} / min: {cosine.min().item():.4f}')

            sim_rewards = self.bleu_factor * bleu_tensor + self.cos_factor * cosine

        elif mode == MrlRewardMode.LONG_RANGE:
            bleu = self.batch_bleu(generated, reference, saved_data, is_first_step=prev_data is None)
            cosine = self.batch_cosine(generated, reference, saved_data,
                                       negative_running_mean=prev_data is not None)

            bleu_tensor = self._to_tensor(bleu, device)

            if self.debug_mode >= 1:
                print('--- LONG MODE')
                print(f'--- BLEU:  {bleu_tensor.mean().item():.4f}  / max: {bleu_tensor.max().item():.4f} / min: {bleu_tensor.min().item():.4f}')
                print(f'--- COSINE: {cosine.mean().item():.4f} / max: {cosine.max().item():.4f} / min: {cosine.min().item():.4f}')

            sim_rewards = self.bleu_factor * bleu_tensor + self.cos_factor * cosine

        else:  # NEGATIVE mode
            bleu = self.negative_bleu(generated, reference, saved_data)
            cosine = self.negative_cosine(generated, reference, saved_data)

            bleu_tensor = self._to_tensor(bleu, device)

            if self.debug_mode >= 1:
                print('--- NEGATIVE MODE')
                print(f'--- BLEU:  {bleu_tensor.mean().item():.4f}  / max: {bleu_tensor.max().item():.4f} / min: {bleu_tensor.min().item():.4f}')
                print(f'--- COSINE: {cosine.mean().item():.4f} / max: {cosine.max().item():.4f} / min: {cosine.min().item():.4f}')

            sim_rewards = self.neg_bleu_factor * bleu_tensor + self.neg_cos_factor * cosine

        if self.reward_len:
            len_reward = self.len_reward(generated, reference)

            if self.debug_mode >= 1:
                print(f'--- REWARD LEN: {len_reward.mean().item():.4f} / max: {len_reward.max().item():.4f} / min: {len_reward.min().item():.4f}')

            rewards = self._pre_scale_rewards(sim_rewards + self.len_factor * len_reward) * self.rewards_scale
        else:
            rewards = self._pre_scale_rewards(sim_rewards) * self.rewards_scale

        return rewards


class HybridMrlRewardModel:
    """
    Hybrid Reward Model that combines Memory Retention Rewards with Preference Rewards.

    This allows training models that both:
    1. Learn to use memory effectively (memory retention rewards)
    2. Generate high-quality, preferred responses (preference rewards)

    The combination can be configured with different weights and modes.

    Args:
        memory_reward_model: Base MrlRewardModel for memory retention rewards
        preference_model: Optional preference model (e.g., reward model trained on preferences)
        preference_weight: Weight for preference rewards (0.0 to 1.0)
        memory_weight: Weight for memory rewards (0.0 to 1.0, should sum to 1.0 with preference_weight)
        mode: How to combine rewards (MEMORY_ONLY, PREFERENCE_ONLY, COMBINED)
        preference_scale: Scale factor for preference rewards
        use_length_penalty: Whether to apply length penalty in preference scoring
        length_penalty_alpha: Alpha parameter for length penalty
    """

    def __init__(
            self,
            memory_reward_model: MrlRewardModel,
            preference_model: Optional[nn.Module] = None,
            preference_weight: float = 0.3,
            memory_weight: float = 0.7,
            mode: PreferenceRewardMode = PreferenceRewardMode.MEMORY_ONLY,
            preference_scale: float = 1.0,
            use_length_penalty: bool = False,
            length_penalty_alpha: float = 0.5,
            debug_mode: int = 0,
    ):
        self.memory_reward_model = memory_reward_model
        self.preference_model = preference_model
        self.preference_weight = preference_weight
        self.memory_weight = memory_weight
        self.mode = mode
        self.preference_scale = preference_scale
        self.use_length_penalty = use_length_penalty
        self.length_penalty_alpha = length_penalty_alpha
        self.debug_mode = debug_mode

        # Validate weights
        if mode == PreferenceRewardMode.COMBINED:
            if abs(preference_weight + memory_weight - 1.0) > 0.01:
                raise ValueError(f"Weights should sum to 1.0 in COMBINED mode, got {preference_weight + memory_weight}")
            if preference_model is None:
                raise ValueError("preference_model is required when mode is COMBINED or PREFERENCE_ONLY")

    def _compute_preference_reward(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
    ) -> torch.Tensor:
        """
        Compute preference reward using the preference model.

        The preference model should output a scalar reward for each generated sequence.
        """
        if self.preference_model is None:
            return torch.zeros(generated['input_ids'].size(0), device=generated['input_ids'].device)

        with torch.no_grad():
            # Get preference scores from model
            # Assuming preference model takes input_ids and attention_mask and outputs scores
            pref_scores = self.preference_model(
                generated['input_ids'],
                attention_mask=generated['attention_mask']
            ).squeeze(-1)

            # Apply length penalty if configured
            if self.use_length_penalty:
                lengths = generated['attention_mask'].sum(dim=-1).float()
                length_penalty = (lengths / lengths.max()) ** self.length_penalty_alpha
                pref_scores = pref_scores * length_penalty

            return pref_scores * self.preference_scale

    def _compute_contrastive_preference_reward(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
    ) -> torch.Tensor:
        """
        Compute contrastive preference reward comparing generated vs reference.

        This gives higher rewards when generated responses are preferred over references,
        which is useful for teaching the model to generate better-than-reference responses.
        """
        if self.preference_model is None:
            return torch.zeros(generated['input_ids'].size(0), device=generated['input_ids'].device)

        with torch.no_grad():
            # Score generated responses
            gen_scores = self.preference_model(
                generated['input_ids'],
                attention_mask=generated['attention_mask']
            ).squeeze(-1)

            # Score reference responses
            ref_scores = self.preference_model(
                reference['input_ids'],
                attention_mask=reference['attention_mask']
            ).squeeze(-1)

            # Contrastive reward: sigmoid of difference
            # This gives rewards in (0, 1) range
            contrast = torch.sigmoid(gen_scores - ref_scores)

            return contrast * self.preference_scale

    def reset_running_mean(self):
        """Reset running mean in memory reward model."""
        self.memory_reward_model.reset_running_mean()

    def init_running_mean(self, prev_data: TokenizedDict):
        """Initialize running mean in memory reward model."""
        self.memory_reward_model.init_running_mean(prev_data)

    def update_running_mean(self, prev_data: TokenizedDict):
        """Update running mean in memory reward model."""
        self.memory_reward_model.update_running_mean(prev_data)

    def __call__(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
            saved_data: TokenizedDict,
            prev_data: TokenizedDict = None,
            mode: MrlRewardMode = MrlRewardMode.STANDARD,
            use_contrastive_preference: bool = False,
    ) -> torch.Tensor:
        """
        Compute hybrid rewards combining memory and preference signals.

        Args:
            generated: Generated token sequences
            reference: Reference answer sequences
            saved_data: Saved memory data sequences
            prev_data: Previous interaction data (for running mean)
            mode: Memory reward mode (STANDARD, NEGATIVE, LONG_RANGE)
            use_contrastive_preference: Whether to use contrastive preference scoring

        Returns:
            Combined reward tensor [batch_size]
        """
        device = generated['input_ids'].device

        if self.mode == PreferenceRewardMode.MEMORY_ONLY:
            # Only memory retention rewards
            return self.memory_reward_model(
                generated, reference, saved_data, prev_data, mode
            )

        elif self.mode == PreferenceRewardMode.PREFERENCE_ONLY:
            # Only preference rewards
            if use_contrastive_preference:
                return self._compute_contrastive_preference_reward(generated, reference)
            else:
                return self._compute_preference_reward(generated, reference)

        else:  # COMBINED mode
            # Compute memory rewards
            memory_rewards = self.memory_reward_model(
                generated, reference, saved_data, prev_data, mode
            )

            # Compute preference rewards
            if use_contrastive_preference:
                preference_rewards = self._compute_contrastive_preference_reward(generated, reference)
            else:
                preference_rewards = self._compute_preference_reward(generated, reference)

            # Combine rewards
            combined_rewards = (
                self.memory_weight * memory_rewards +
                self.preference_weight * preference_rewards
            )

            if self.debug_mode >= 1:
                print(f'--- HYBRID REWARDS ---')
                print(f'--- Memory: mean={memory_rewards.mean().item():.4f}, '
                      f'max={memory_rewards.max().item():.4f}, min={memory_rewards.min().item():.4f}')
                print(f'--- Preference: mean={preference_rewards.mean().item():.4f}, '
                      f'max={preference_rewards.max().item():.4f}, min={preference_rewards.min().item():.4f}')
                print(f'--- Combined: mean={combined_rewards.mean().item():.4f}, '
                      f'max={combined_rewards.max().item():.4f}, min={combined_rewards.min().item():.4f}')

            return combined_rewards
