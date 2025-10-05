import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Optional, Literal, TypeAlias, Union
from ..training.utils import TokenizedDict
from ..metrics.tensorbleu import tensor_sentence_bleu


SimpleMrqBenchScoreDict: TypeAlias = dict[
    Literal['bleu', 'cosine', 'length', 'raw', 'score'], dict[
        Literal['all', 'mean', 'max', 'min'], torch.Tensor
    ]
]

class SimpleMrqBenchScore:
    def __init__(
            self,
            shared_embedding: nn.Embedding,
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
            allow_not_summing_factors: bool = False,
            neg_reward_len: bool = False,
            max_rewarded_len: int = None,
            target_len_as_ref: bool = False,
            len_factor: int = None,
            running_mean_decay: float = 0.2,
            bleu_saved_weights: tuple = (0.5, 0.5),
            bleu_ref_weights: tuple = (0.5, 0.5),
            rewards_scale: float = 1.0,
            debug_mode: int = 0,
            use_tensor_bleu: bool = False,
            tensor_bleu_smoothing: Literal['none', 'floor', 'add-k', 'exp'] = 'exp',
    ):
        self.shared_embedding = shared_embedding
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
        self.neg_reward_len = neg_reward_len
        self.max_rewarded_len = max_rewarded_len
        self.target_len_as_ref = target_len_as_ref
        self.len_factor = len_factor
        self.running_mean_decay = running_mean_decay
        self.bleu_ref_weights = bleu_ref_weights
        self.bleu_saved_weights = bleu_saved_weights
        self.rewards_scale = rewards_scale
        self.bleu_smoothing = SmoothingFunction().method4
        self.debug_mode = debug_mode
        self.use_tensor_bleu = use_tensor_bleu
        self.tensor_bleu_smoothing = tensor_bleu_smoothing

        self.prev_data_running_mean = None

        if not allow_not_summing_factors:
            assert self.bleu_factor + self.cos_factor + self.len_factor == 1.0
            assert self.multi_cos_ref_factor + self.multi_cos_saved_factor + self.multi_cos_running_mean_factor == 1.0
            assert self.bleu_ref_factor + self.bleu_saved_factor == 1.0
            assert self.cos_ref_factor + self.cos_saved_factor == 1.0


    def _sentence_bleu(self, input_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                       masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], is_first_step: bool = False) -> float:
        generated, reference, saved_data = input_ids
        generated_mask, reference_mask, saved_data_mask = masks

        generated = generated.tolist()[:generated_mask.sum().item()]
        reference = reference.tolist()[:reference_mask.sum().item()]
        saved_data = saved_data.tolist()[:saved_data_mask.sum().item()]

        if self.bleu_mode == 'separate':
            ref_bleu = sentence_bleu([reference], generated, weights=self.bleu_ref_weights,
                                     smoothing_function=self.bleu_smoothing)
            saved_bleu = sentence_bleu([saved_data], generated, weights=self.bleu_saved_weights,
                                       smoothing_function=self.bleu_smoothing)

            if is_first_step:
                return self.bleu_first_ref_factor * ref_bleu + self.bleu_first_saved_factor * saved_bleu
            else:
                return self.bleu_ref_factor * ref_bleu + self.bleu_saved_factor * saved_bleu
        else:
            return sentence_bleu(
                [reference, saved_data], generated,
                weights=self.bleu_ref_weights, smoothing_function=self.bleu_smoothing
            )

    def batch_bleu(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict, is_first_step: bool = False) -> Union[list[float], torch.Tensor]:
        if self.use_tensor_bleu:
          return self.batch_tensor_bleu(generated, reference, saved_data, is_first_step)

        batch_size = generated['input_ids'].size(0)

        return [
            self._sentence_bleu(
                input_ids=(generated['input_ids'][i], reference['input_ids'][i], saved_data['input_ids'][i]),
                masks=(generated['attention_mask'][i], reference['attention_mask'][i], saved_data['attention_mask'][i]),
                is_first_step=is_first_step,
            ) for i in range(batch_size)
        ]

    def batch_tensor_bleu(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict, is_first_step: bool = False) -> torch.Tensor:
        if self.bleu_mode == 'separate':
            ref_bleu = tensor_sentence_bleu(
                generated['input_ids'], reference['input_ids'].unsqueeze(1),
                weights=torch.tensor(self.bleu_ref_weights), smoothing_method=self.tensor_bleu_smoothing
            )
            saved_bleu = tensor_sentence_bleu(
                generated['input_ids'], saved_data['input_ids'].unsqueeze(1),
                weights=torch.tensor(self.bleu_saved_weights), smoothing_method=self.tensor_bleu_smoothing
            )

            if is_first_step:
                return self.bleu_first_ref_factor * ref_bleu + self.bleu_first_saved_factor * saved_bleu
            else:
                return self.bleu_ref_factor * ref_bleu + self.bleu_saved_factor * saved_bleu
        else:
            device = generated['input_ids'].device
            return tensor_sentence_bleu(
                generated['input_ids'], torch.tensor([reference['input_ids'], saved_data['input_ids']]).to(device),
                weights=torch.tensor(self.bleu_ref_weights), smoothing_method=self.tensor_bleu_smoothing
            )

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

        return torch.clamp(gen_and_saved, min=0), torch.clamp(gen_and_ref, min=0)

    def _cosine_sim_running_mean(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict):
        generated_emb = F.normalize(self._sequence_embedding(generated), dim=-1)
        saved_data_emb = F.normalize(self._sequence_embedding(saved_data), dim=-1)
        reference_emb = F.normalize(self._sequence_embedding(reference), dim=-1)
        running_emb = F.normalize(self.prev_data_running_mean, dim=-1)

        gen_and_saved = F.cosine_similarity(generated_emb, saved_data_emb, dim=1)
        gen_and_ref = F.cosine_similarity(generated_emb, reference_emb, dim=1)
        gen_and_mean = F.cosine_similarity(generated_emb, running_emb, dim=1)

        return torch.clamp(gen_and_saved, min=0), torch.clamp(gen_and_ref, min=0), torch.clamp(gen_and_mean, min=0)

    def batch_cosine(self, generated: TokenizedDict, reference: TokenizedDict, saved_data: TokenizedDict,
                     include_running_mean: bool = False) -> torch.Tensor:
        if include_running_mean:
            gen_and_saved, gen_and_ref, gen_and_mean = self._cosine_sim_running_mean(generated, reference, saved_data)
            return self.multi_cos_saved_factor * gen_and_saved + self.multi_cos_ref_factor * gen_and_ref + self.multi_cos_running_mean_factor * gen_and_mean
        else:
            gen_and_saved, gen_and_ref = self._cosine_sim(generated, reference, saved_data)
            return self.cos_saved_factor * gen_and_saved + self.cos_ref_factor * gen_and_ref

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

    def __call__(
            self,
            generated: TokenizedDict,
            reference: TokenizedDict,
            saved_data: TokenizedDict,
            prev_data: TokenizedDict = None,
    ) -> SimpleMrqBenchScoreDict:
        if prev_data is not None:
            if self.prev_data_running_mean is None:
                self.init_running_mean(prev_data)
            else:
                self.update_running_mean(prev_data)

        device = generated['input_ids'].device

        bleu = self.batch_bleu(generated, reference, saved_data, is_first_step=prev_data is None)
        cosine = self.batch_cosine(generated, reference, saved_data, include_running_mean=prev_data is not None)
        len_reward = self.len_reward(generated, reference)

        tensor_bleu = torch.tensor(bleu, device=device) if not self.use_tensor_bleu else bleu

        raw_score = self.bleu_factor * tensor_bleu + self.cos_factor * cosine + self.len_factor * len_reward
        scaled_score = raw_score * self.rewards_scale

        return {
            'bleu': {
                'all': tensor_bleu,
                'mean': tensor_bleu.mean(),
                'max': tensor_bleu.max(),
                'min': tensor_bleu.min(),
            },
            'cosine': {
                'all': cosine,
                'mean': cosine.mean(),
                'max': cosine.max(),
                'min': cosine.min(),
            },
            'length': {
                'all': len_reward,
                'mean': len_reward.mean(),
                'max': len_reward.max(),
                'min': len_reward.min(),
            },
            'raw': {
                'all': raw_score,
                'mean': raw_score.mean(),
                'max': raw_score.max().item(),
                'min': raw_score.min().item(),
            },
            'score': {
                'all': scaled_score,
                'mean': scaled_score.mean(),
                'max': scaled_score.max(),
                'min': scaled_score.min(),
            }
        }
