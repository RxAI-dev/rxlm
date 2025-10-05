import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, TypedDict, Union, Literal
from ..llm.models import DecoderOnlyTransformer
from ..training.utils import smart_concat, TokenizedDict
from .score import SimpleMrqBenchScore, SimpleMrqBenchScoreDict
from .models import RxTMrqBenchModel, RxTMrqBenchAction
from .dataset import RxlmMrqBenchDataset, LlmMrqBenchDataset
from .sampler import MrqBenchBatchSampler

class MrqBenchTokenizerConfig(TypedDict):
    pad_token_id: int
    end_token_id: int
    answer_token_id: int


class SamplerConfig(TypedDict):
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]


class MrqBenchRunner:
    def __init__(
            self,
            device: torch.device,
            score: SimpleMrqBenchScore,
            max_seq_len: int,
            interaction_len: int,
            tokenizer_config: MrqBenchTokenizerConfig,
            sampler_config: Optional[SamplerConfig] = None,
            use_amp: bool = False,
            dtype: torch.dtype = torch.float32,
            teacher_forcing: bool = True,
    ):
        self.score = score
        self.device = device

        # Batch Sampler for answer generation
        self.generator = None
        self.sampler_config = SamplerConfig(
            temperature=1.0,
            top_k=None,
            top_p=None,
        ) if sampler_config is None else sampler_config

        self.pad_token_id = tokenizer_config.get('pad_token_id', 0)
        self.end_token_id = tokenizer_config.get('end_token_id', 3)
        self.answer_token_id = tokenizer_config.get('answer_token_id', 6)

        self.use_amp = use_amp
        self.dtype = dtype

        self.max_seq_len = max_seq_len
        self.interaction_len = interaction_len

        # Dynamic fields, updated for each curriculum step
        self.dataset = None
        self.teacher_forcing = teacher_forcing
        self.model = None


    def encode_and_update_stm(self, query: TokenizedDict, answer: TokenizedDict):
        """Encode interaction and update STM."""
        # 1. Encode data and update memory - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # 2. Concatenate batch of queries and answers (they are already on training device)
                inputs = smart_concat(query, answer, self.interaction_len, self.pad_token_id)
                # 3. Encode data and update STM
                self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=RxTMrqBenchAction.UPDATE)
        else:
            # 2. Concatenate batch of queries and answers (they are already on training device)
            inputs = smart_concat(query, answer, self.interaction_len, self.pad_token_id)
            # 3. Encode data and update STM
            self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=RxTMrqBenchAction.UPDATE)

    def generate_answer(self, query: TokenizedDict, timing_log: list[dict[str, float]]) -> TokenizedDict:
        """Generate response using batch sampler with decoder."""
        # 1. Generate answer with BatchSampler - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                input_ids, attention_mask, _ = self.generator(
                    query['input_ids'],
                    query['attention_mask'],
                    max_gen_len=self.interaction_len,
                    dtype=self.dtype,
                    timing_log=timing_log,
                    **self.sampler_config,
                )
        else:
            input_ids, attention_mask, _ = self.generator(
                query['input_ids'],
                query['attention_mask'],
                max_gen_len=self.interaction_len,
                dtype=self.dtype,
                timing_log=timing_log,
                **self.sampler_config,
            )
        # 2. Convert generated answer to TokenizedDict
        generated_answer: TokenizedDict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        return generated_answer

    def compute_score(
            self, generated: TokenizedDict, reference: TokenizedDict, saved_data: tuple[TokenizedDict, TokenizedDict],
            prev_data: tuple[TokenizedDict, TokenizedDict] = None
    ) -> SimpleMrqBenchScoreDict:
        """Compute reward based on memory retention (e.g., BLEU-4)."""
        # 1. Move sequences to GPU for reward calculation
        saved_query, saved_answer = self._move_multiple_batches(*saved_data)
        reference = self._move_batch(reference)
        prev_data = self._move_multiple_batches(*prev_data) if prev_data is not None else None

        saved_interaction = smart_concat(
            saved_query, saved_answer, max_length=self.interaction_len, pad_token_id=self.pad_token_id
        )
        prev_interaction = smart_concat(prev_data[0], prev_data[1], self.interaction_len,
                                 self.pad_token_id) if prev_data is not None else None

        # 2. Concat saved (previous) interaction and calculate reward using generated sequence, reference and saved data - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                scoring = self.score(generated, reference, saved_interaction, prev_data=prev_interaction)
        else:
            scoring = self.score(generated, reference, saved_interaction, prev_data=prev_interaction)

        # 4. Return scores for batch
        return scoring

    def _move_batch(self, batch: TokenizedDict) -> TokenizedDict:
        if self.use_amp:
            return {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
            }
        else:
            return {
                'input_ids': batch['input_ids'].to(self.device, dtype=self.dtype),
                'attention_mask': batch['attention_mask'].to(self.device, dtype=self.dtype),
            }

    def _move_multiple_batches(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._move_batch(batch) for batch in batches]

    def _cpu_detach(self, batch: TokenizedDict) -> TokenizedDict:
        return {
            'input_ids': batch['input_ids'].detach().cpu(),
            'attention_mask': batch['attention_mask'].detach().cpu(),
        }

    def _batch_detach(self, batch: TokenizedDict) -> TokenizedDict:
        return {
            'input_ids': batch['input_ids'].detach(),
            'attention_mask': batch['attention_mask'].detach(),
        }

    def _cpu_detach_multiple(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._cpu_detach(batch) for batch in batches]

    def _rxlm_loader(self, batch_size: int):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=RxlmMrqBenchDataset.collate_mrl_batch,
        )

    def _llm_loader(self, batch_size: int):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=LlmMrqBenchDataset.collate_llm_batch,
        )

    def run_rxlm(self, batch_size: int, num_of_examples: int = None):
        # 1. Init evaluation DataLoader
        dataloader = self._rxlm_loader(batch_size)
        self.model.eval()

        results = []

        # 2. Run evaluation on all batch episodes
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                if num_of_examples is not None:
                    if batch_idx >= num_of_examples:
                        break
                if batch['query']['input_ids'].size(0) == batch_size:
                    # 3. Reset STM with random resets ratio and reward model running mean
                    self.model.reset_memory()
                    self.score.reset_running_mean()

                    # 4. Get batches for first queries, answers and all follow-up interactions
                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                    # 5. Encode and update STM with initial interactions (batch)
                    first_interaction = self._move_multiple_batches(first_query, first_answer)
                    self.encode_and_update_stm(*first_interaction)

                    # 6. Save follow-up interactions len and first query and answer as previous one for iteration
                    interactions_len = len(interactions)
                    query, answer = first_query, first_answer

                    prev_interaction = None

                    episode_results: list[SimpleMrqBenchScoreDict] = []
                    timing_log: list[dict[str, float]] = []

                    # 7. Run all follow-up interactions
                    for i, interaction in enumerate(interactions):
                        # 8. Generate batch of answers
                        next_query = self._move_batch(interaction['query'])
                        generated_answer = self.generate_answer(next_query, timing_log)

                        is_last_interaction = (i + 1) == interactions_len

                        detached_answer = self._batch_detach(generated_answer)

                        # 9. Depending on current strategy and step, compute reward
                        scoring = self.compute_score(
                            detached_answer, interaction['answer'], (query, answer), prev_data=prev_interaction
                        )

                        print(f'Step: {i + 1}. Mean reward: {scoring['score']['mean'].item()} / Min reward: {scoring['score']['min'].item()} / Max reward: {scoring['score']['max'].item()}')
                        print(f'Step: {i + 1}. Mean BLEU: {scoring['bleu']['mean'].item()} Mean Cosine: {scoring['cosine']['mean'].item()}')

                        episode_results.append(scoring)

                        # 10. Encode and update memory for the next interaction
                        if not is_last_interaction:
                            self.encode_and_update_stm(
                                next_query,
                                self._move_batch(interaction['answer']) if self.teacher_forcing else generated_answer
                            )

                        # 12. Save previous interaction
                        prev_interaction = (query, answer)
                        query, answer = interaction['query'], (interaction['answer'] if self.teacher_forcing else self._cpu_detach(generated_answer))

                    batch_mean_scores = torch.stack([item['score']['all'] for item in episode_results]).transpose(0, 1).mean(dim=-1)

                    episode = {
                        'scores': episode_results,
                        'mean_score': torch.stack([item['score']['mean'] for item in episode_results]).mean(),
                        'all': batch_mean_scores,
                        'mean': batch_mean_scores.mean(),
                        'max': batch_mean_scores.max(),
                        'min': batch_mean_scores.min(),
                        'timing': timing_log,
                    }
                    results.append(episode)

                    print(
                        f'Mean episode reward: {episode["mean_score"].item()} | {episode["mean"].item()} / Min episode reward: {episode["min"].item()} / Max episode reward: {episode["max"].item()}')

        # 15. Calculate average reward
        avg_score = torch.stack([result['mean_score'] for result in results]).mean() if len(results) > 0 else torch.tensor(0.0)

        return {
            'mean': avg_score.item(),
            'scores': results,
        }

    def run_llm(self, batch_size: int, num_of_examples: int = None):
        # 1. Init evaluation DataLoader
        dataloader = self._llm_loader(batch_size)
        self.model.eval()

        results = []

        # 2. Run evaluation on all batch episodes
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                if num_of_examples is not None:
                    if batch_idx >= num_of_examples:
                        break
                if batch['query']['input_ids'].size(0) == batch_size:
                    self.score.reset_running_mean()

                    # 4. Get batches for first queries, answers and all follow-up interactions
                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']

                    # 6. Save follow-up interactions len and first query and answer as previous one for iteration
                    query, answer = first_query, first_answer

                    prev_interaction = None

                    episode_results: list[SimpleMrqBenchScoreDict] = []
                    timing_log: list[dict[str, float]] = []

                    # 7. Run all follow-up interactions
                    for i, interaction in enumerate(interactions):
                        # 8. Generate batch of answers
                        next_context = self._move_batch(interaction['context'])
                        generated_answer = self.generate_answer(next_context, timing_log)

                        detached_answer = self._batch_detach(generated_answer)

                        # 9. Depending on current strategy and step, compute reward
                        scoring = self.compute_score(
                            detached_answer, interaction['answer'], (query, answer), prev_data=prev_interaction
                        )

                        print(f'Step: {i + 1}. Mean reward: {scoring['score']['mean'].item()} / Min reward: {scoring['score']['min'].item()} / Max reward: {scoring['score']['max'].item()}')
                        print(f'Step: {i + 1}. Mean BLEU: {scoring['bleu']['mean'].item()} Mean Cosine: {scoring['cosine']['mean'].item()}')

                        episode_results.append(scoring)
                        # 12. Save previous interaction
                        prev_interaction = (query, answer)
                        query, answer = interaction['query'], (
                            interaction['answer'] if self.teacher_forcing else self._cpu_detach(generated_answer))

                    batch_mean_scores = torch.stack([item['score']['all'] for item in episode_results]).transpose(0, 1).mean(dim=-1)

                    episode = {
                        'scores': episode_results,
                        'mean_score': torch.stack([item['score']['mean'] for item in episode_results]).mean(),
                        'all': batch_mean_scores,
                        'mean': batch_mean_scores.mean(),
                        'max': batch_mean_scores.max(),
                        'min': batch_mean_scores.min(),
                        'timing': timing_log,
                    }
                    results.append(episode)

                    print(f'Mean episode reward: {episode["mean_score"].item()} | {episode["mean"].item()} / Min episode reward: {episode["min"].item()} / Max episode reward: {episode["max"].item()}')


        # 15. Calculate average reward
        avg_score = torch.stack([result['mean_score'] for result in results]).mean() if len(results) > 0 else torch.tensor(0.0)

        return {
            'mean': avg_score.item(),
            'scores': results,
        }


    def __call__(self, mode: Literal['rxlm', 'llm'], model: Union[DecoderOnlyTransformer, RxTMrqBenchModel], dataset: Union[RxlmMrqBenchDataset, LlmMrqBenchDataset], batch_size: int, num_of_examples: int = None):
        self.model = model
        self.dataset = dataset

        self.generator = MrqBenchBatchSampler(
            self.model, self.device, end_token_id=self.end_token_id, answer_token_id=self.answer_token_id,
            pad_token_id=self.pad_token_id, use_self_attn_cache=True, use_stm_kv_cache=mode=='rxlm', use_first_empty_workaround=False
        )

        if mode == 'rxlm':
            scores = self.run_rxlm(batch_size, num_of_examples)
        else:
            scores = self.run_llm(batch_size, num_of_examples)

        return scores
