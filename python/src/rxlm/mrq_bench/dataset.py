import torch
from torch.utils.data import Dataset
from datasets import Dataset as HfDataset, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union, Optional, TypeAlias, Any, Literal
from ..training.dataset import MrlCurriculumDataset


class RxlmMrqBenchDataset(MrlCurriculumDataset):
    pass


ItemFields: TypeAlias = Literal['input_ids', 'attention_mask']
LlmMrqBenchDatasetItem: TypeAlias = dict[str, Union[dict[str, torch.Tensor], list[dict[str, dict[str, torch.Tensor]]]]]


class LlmMrqBenchDataset(Dataset):
    def __init__(
            self,
            episodes: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 2048,
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            interaction_len: Optional[int] = None,
            **kwargs,
    ):
        super(LlmMrqBenchDataset, self).__init__(**kwargs)
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.interaction_len = interaction_len if interaction_len is not None else max_seq_len
        self.query_field = query_field
        self.answer_field = answer_field
        self.interactions_field = interactions_field
        self.query_token = query_token
        self.answer_token = answer_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.is_pre_tokenized = False
        self.is_list = isinstance(self.episodes, list)
        self.inputs = []

    def _tokenize_manual_interaction(self, query: str, answer: str) -> dict[str, dict[str, torch.Tensor]]:
        # Manually construct query: [BOS][Q]query
        query_text = f"{self.bos_token}{self.query_token}{query}"
        query_enc = self.tokenizer(
            query_text,
            max_length=self.interaction_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Critical: We control all tokens
        )

        # Manually construct answer: [A]answer[EOS]
        answer_text = f"{self.answer_token}{answer}{self.eos_token}"
        answer_enc = self.tokenizer(
            answer_text,
            max_length=self.interaction_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Critical: We control all tokens
        )

        return {
            'query': {
                'input_ids': query_enc['input_ids'][0],
                'attention_mask': query_enc['attention_mask'][0],
            },
            'answer': {
                'input_ids': answer_enc['input_ids'][0],
                'attention_mask': answer_enc['attention_mask'][0],
            }
        }

    def _tokenize_template(self, initial: tuple[str, str], interactions: list[dict[str, str]], idx: int) -> dict[str, dict[str, torch.Tensor]]:
        init_query, init_answer = initial
        ctx_txt = f"{self.bos_token}{self.query_token}{init_query}{self.answer_token}{init_answer}"

        for i in range(idx):
            item = interactions[i]
            query = item[self.query_field]
            answer = item[self.answer_field]
            ctx_txt += f"{self.query_token}{query}{self.answer_token}{answer}"

        current_item = interactions[idx]
        current_query = current_item[self.query_field]
        current_answer = current_item[self.answer_field]

        ctx_txt += f"{self.query_token}{current_query}"

        ctx_enc = self.tokenizer(
            ctx_txt,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Critical: We control all tokens
        )

        return {
            'context': {
                'input_ids': ctx_enc['input_ids'][0],
                'attention_mask': ctx_enc['attention_mask'][0],
            },
            **self._tokenize_manual_interaction(current_query, current_answer),
        }

    def get_tokenized_item(self, idx: int, episode: dict = None) -> LlmMrqBenchDatasetItem:
        if self.is_pre_tokenized:
            return self.inputs[idx]
        else:
            item = self.episodes[idx] if episode is None else episode

            query = item[self.query_field]
            answer = item[self.answer_field]
            interactions = item[self.interactions_field]

            initial = self._tokenize_manual_interaction(query, answer)
            templates = [self._tokenize_template((query, answer), interactions, idx) for idx in range(len(interactions))]

            return {
                **initial,
                'interactions': templates,
            }

    def __getitem__(self, idx: int) -> LlmMrqBenchDatasetItem:
        return self.get_tokenized_item(idx)

    def __len__(self) -> int:
        return len(self.inputs if self.is_pre_tokenized else self.episodes)

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "LlmMrqBenchDataset":
        split_point = int(len(self.inputs if self.is_pre_tokenized else self.episodes) * ((1 - size) if not from_start else size))
        if not isinstance(self.episodes, list):
            subset = self.episodes.select(
                range(split_point, len(self.episodes)) if not from_start else range(split_point))
            self.episodes = self.episodes.select(
                range(split_point) if not from_start else range(split_point, len(self.episodes)))
        else:
            subset = self.episodes[split_point:-1] if not from_start else self.episodes[0:split_point]
            self.episodes = self.episodes[0:split_point] if not from_start else self.episodes[split_point:-1]
        return self.__class__(subset, tokenizer=self.tokenizer, query_field=self.query_field,
                              answer_field=self.answer_field, interactions_field=self.interactions_field,
                              max_seq_len=self.max_seq_len, **kwargs)

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000, keep_order: bool = False):
        """
        Pre-tokenizes all the items in the dataset, for faster training. Training with pre-tokenized
        dataset could be even 2x faster.

        !! This method has extremely high memory usage, when used with HuggingFace datasets,
        because of converting it to list. Additionally, for the most optimal performance,
        pre-tokenized items are in reversed order - it shouldn't matter for training, as
        items are shuffled then by DataLoader, but you should keep that in mind in case
        of reproducibility.

        Args:
            verbose (bool): Should display logs (default: False)
            log_interval (int): Display logs every log_interval iterations (default: 10_000)
            keep_order (bool): Keep tokenized items in the same order - by default they are reversed for faster processing (default: False)
        """
        if not self.is_pre_tokenized:
            num_episodes = len(self.episodes)
            eps = self.episodes if self.is_list else self.episodes.to_list()
            del self.episodes
            self.episodes = None
            for index in range(num_episodes):
                self.inputs.append(self.get_tokenized_item(index, episode=eps.pop() if not keep_order else eps[index]))
                if verbose and index % log_interval == 0:
                    print(f'Processed {index + 1}/{num_episodes}')
            del eps
            self.is_pre_tokenized = True

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            mrl_subset: str,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            split: str = 'train',
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            load_kwargs: dict = None,
            max_seq_len: int = 2048,
            **kwargs
    ):
        """
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        Args:
            dataset_id (str): Hub dataset repository name
            mrl_subset (str): Dataset subset
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer
            split (str): Dataset split (default: "train")
            query_field (str): Query field (default: "query")
            answer_field (str): Answer field (default: "answer")
            interactions_field (str): Interactions field (default: "interactions")
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            max_seq_len (int): Maximum sequence length (default: 2048)
            **kwargs: Additional args for RxNN Dataset class
        """
        if load_kwargs is None:
            load_kwargs = {}

        hf_dataset = load_dataset(dataset_id, mrl_subset, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, query_field=query_field, answer_field=answer_field,
                   interactions_field=interactions_field, max_seq_len=max_seq_len, **kwargs)

    @staticmethod
    def collate_llm_batch(batch: list[LlmMrqBenchDatasetItem]) -> LlmMrqBenchDatasetItem:
        """Collate function for MRL curriculum dataset with nested interactions"""

        def collate_first_interaction_batch(interaction_batch: Union[list[dict[str, dict[str, torch.Tensor]]], tuple[Any]]) -> \
        dict[str, dict[ItemFields, torch.Tensor]]:
            """Helper to collate a batch of interactions"""
            return {
                'query': {
                    'input_ids': torch.stack([x['query']['input_ids'] for x in interaction_batch]),
                    'attention_mask': torch.stack([x['query']['attention_mask'] for x in interaction_batch]),
                },
                'answer': {
                    'input_ids': torch.stack([x['answer']['input_ids'] for x in interaction_batch]),
                    'attention_mask': torch.stack([x['answer']['attention_mask'] for x in interaction_batch]),
                }
            }

        def collate_interaction_batch(interaction_batch: Union[list[dict[str, dict[str, torch.Tensor]]], tuple[Any]]) -> \
        dict[str, dict[ItemFields, torch.Tensor]]:
            """Helper to collate a batch of interactions"""
            return {
                'context': {
                    'input_ids': torch.stack([x['context']['input_ids'] for x in interaction_batch]),
                    'attention_mask': torch.stack([x['context']['attention_mask'] for x in interaction_batch]),
                },
                'query': {
                    'input_ids': torch.stack([x['query']['input_ids'] for x in interaction_batch]),
                    'attention_mask': torch.stack([x['query']['attention_mask'] for x in interaction_batch]),
                },
                'answer': {
                    'input_ids': torch.stack([x['answer']['input_ids'] for x in interaction_batch]),
                    'attention_mask': torch.stack([x['answer']['attention_mask'] for x in interaction_batch]),
                }
            }

        batch_interactions = [x['interactions'] for x in batch]
        transposed_interactions = list(zip(*batch_interactions))

        return {
            **collate_first_interaction_batch(batch),  # Collate initial query and answer
            'interactions': [
                collate_interaction_batch(step_batch) for step_batch in transposed_interactions
            ]
        }
