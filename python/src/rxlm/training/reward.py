import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from enum import Enum
from typing import Optional, Literal
from .utils import TokenizedDict


class MrlRewardMode(Enum):
    STANDARD = 1
    NEGATIVE = 2
    LONG_RANGE = 3


class MrlRewardModel:
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
        self.bleu_smoothing = SmoothingFunction().method4
        self.debug_mode = debug_mode

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

    def batch_bleu(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict, is_first_step: bool = False) -> list[float]:
        batch_size = generated['input_ids'].size(0)

        return [
            self._sentence_bleu(
                input_ids=(generated['input_ids'][i], reference['input_ids'][i], saved_data['input_ids'][i]),
                masks=(generated['attention_mask'][i], reference['attention_mask'][i], saved_data['attention_mask'][i]),
                is_first_step=is_first_step,
            ) for i in range(batch_size)
        ]

    def negative_bleu(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict) -> list[
        float]:
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

    def __call__(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
            saved_data: TokenizedDict,
            prev_data: TokenizedDict = None,
            mode: MrlRewardMode = MrlRewardMode.STANDARD
    ) -> torch.Tensor:
        if prev_data is not None:
            if self.prev_data_running_mean is None:
                self.init_running_mean(prev_data)
            else:
                self.update_running_mean(prev_data)

        device = generated['input_ids'].device

        if mode == MrlRewardMode.STANDARD:
            bleu = self.batch_bleu(generated, reference, saved_data, is_first_step=prev_data is None)
            cosine = self.batch_cosine(generated, reference, saved_data, include_running_mean=prev_data is not None)

            if self.debug_mode >= 1:
                print('--- STANDARD MODE')
                print(f'--- BLEU:  {sum(bleu) / len(bleu)}  / max: {max(bleu)} / min: {min(bleu)}')
                print(f'--- COSINE: {sum(cosine) / len(cosine)} / max: {max(cosine)} / min: {min(cosine)}')

            sim_rewards = self.bleu_factor * torch.tensor(bleu, device=device) + self.cos_factor * cosine
        elif mode == MrlRewardMode.LONG_RANGE:
            bleu = self.batch_bleu(generated, reference, saved_data, is_first_step=prev_data is None)
            cosine = self.batch_cosine(generated, reference, saved_data,
                                       negative_running_mean=prev_data is not None)

            if self.debug_mode >= 1:
                print('--- LONG MODE')
                print(f'--- BLEU:  {sum(bleu) / len(bleu)}  / max: {max(bleu)} / min: {min(bleu)}')
                print(f'--- COSINE: {sum(cosine) / len(cosine)} / max: {max(cosine)} / min: {min(cosine)}')

            sim_rewards = self.bleu_factor * torch.tensor(bleu, device=device) + self.cos_factor * cosine
        else:
            bleu = self.negative_bleu(generated, reference, saved_data)
            cosine = self.negative_cosine(generated, reference, saved_data)

            if self.debug_mode >= 1:
                print('--- NEGATIVE MODE')
                print(f'--- BLEU:  {sum(bleu) / len(bleu)}  / max: {max(bleu)} / min: {min(bleu)}')
                print(f'--- COSINE: {sum(cosine) / len(cosine)} / max: {max(cosine)} / min: {min(cosine)}')

            sim_rewards = self.neg_bleu_factor * torch.tensor(bleu, device=device) + self.neg_cos_factor * cosine

        if self.reward_len:
            len_reward = self.len_reward(generated, reference)

            if self.debug_mode >= 1:
                print(f'--- REWARD LEN: {(len_reward.sum() / len_reward.size(0)).item()} / max: {len_reward.max().item()} / min: {len_reward.min().item()}')

            rewards = self._pre_scale_rewards(sim_rewards + self.len_factor * len_reward) * self.rewards_scale
        else:
            rewards = self._pre_scale_rewards(sim_rewards) * self.rewards_scale

        return rewards
