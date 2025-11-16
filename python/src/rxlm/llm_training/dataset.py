import torch
from datasets import load_dataset, Dataset as HfDataset, IterableDataset as HfIterableDataset
from torch.utils.data import Dataset, get_worker_info
import torch.distributed as dist
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union, TypeAlias, Literal
from ..training.dataset import BaseDataset, BaseIterableDataset
from ..training.tokenizer import load_tokenizer_from_hf_hub

class MaskedLMDataset(BaseDataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            mask_prob: float = 0.15,
            hf_field: str = 'text',
            *args,
            **kwargs
    ):
        super(MaskedLMDataset, self).__init__(texts, tokenizer, max_seq_len, hf_field, *args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inputs = self.get_tokenized_text(idx)

        input_ids = inputs['input_ids'][0]
        if self.is_pre_tokenized:
            input_ids = input_ids.clone()
        attention_mask = inputs['attention_mask'][0]
        labels = input_ids.clone()

        # Create masked indices
        masked_indices = torch.bernoulli(
            torch.full(labels.shape, self.mask_prob)
        ).bool() & attention_mask.bool()

        # Apply mask
        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class AutoregressiveLMDataset(BaseDataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            hf_field: str = 'text',
            *args,
            **kwargs
    ):
        super(AutoregressiveLMDataset, self).__init__(texts, tokenizer, max_seq_len, hf_field, *args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inputs = self.get_tokenized_text(idx)

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        targets = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }

ItemFields: TypeAlias = Literal['input_ids', 'attention_mask', 'targets']
SftDataItem: TypeAlias = dict[ItemFields, torch.Tensor]

class DecoderOnlySftDataset(Dataset):
    def __init__(
            self,
            episodes: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            ignore_index: int = -100,
            **kwargs,
    ):
        super(DecoderOnlySftDataset, self).__init__(**kwargs)
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.query_field = query_field
        self.answer_field = answer_field
        self.interactions_field = interactions_field
        self.query_token = query_token
        self.answer_token = answer_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.ignore_index = ignore_index
        self.is_pre_tokenized = False
        self.is_list = isinstance(self.episodes, list)
        self.inputs = []

    def _build_full_conversation_text(self, item: dict) -> str:
        full_text = f"{self.bos_token}{self.query_token}{item[self.query_field]}{self.answer_token}{item[self.answer_field]}"

        for inter in item[self.interactions_field]:
            full_text += f"{self.query_token}{inter[self.query_field]}{self.answer_token}{inter[self.answer_field]}"

        full_text += self.eos_token
        return full_text

    def _create_masked_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()

        query_token_id = self.tokenizer.convert_tokens_to_ids(self.query_token)
        answer_token_id = self.tokenizer.convert_tokens_to_ids(self.answer_token)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

        in_answer = False
        for i in range(len(input_ids)):
            token = input_ids[i].item()
            if token == answer_token_id:
                in_answer = True
            elif token in (query_token_id, eos_token_id):
                in_answer = False

            if not in_answer or token == answer_token_id:
                labels[i] = self.ignore_index

        return labels

    def get_tokenized_item(self, idx: int, episode: dict = None) -> SftDataItem:
        if self.is_pre_tokenized:
            return self.inputs[idx]
        else:
            item = self.episodes[idx] if episode is None else episode
            full_text = self._build_full_conversation_text(item)

            enc = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False
            )

            input_ids = enc['input_ids'][0]
            attention_mask = enc['attention_mask'][0]
            targets = self._create_masked_labels(input_ids)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'targets': targets,
            }

    def __getitem__(self, idx: int) -> SftDataItem:
        return self.get_tokenized_item(idx)

    def __len__(self) -> int:
        return len(self.inputs if self.is_pre_tokenized else self.episodes)

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "DecoderOnlySftDataset":
        split_point = int(
            len(self.inputs if self.is_pre_tokenized else self.episodes) * ((1 - size) if not from_start else size))
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
                              max_seq_len=self.max_seq_len, ignore_index=self.ignore_index, **kwargs)

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
            max_seq_len: int = 1024,
            ignore_index: int = -100,
            **kwargs
    ):
        """
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_id (str): Hub dataset repository name
            mrl_subset (str): Dataset subset
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer
            split (str): Dataset split (default: "train")
            query_field (str): Query field (default: "query")
            answer_field (str): Answer field (default: "answer")
            interactions_field (str): Interactions field (default: "interactions")
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            max_seq_len (int): Maximum sequence length (default: 1024)
            ignore_index (int): Ignore index for labels masking (default: -100)
            **kwargs: Additional args for RxNN Dataset class
        """
        if load_kwargs is None:
            load_kwargs = {}

        hf_dataset = load_dataset(dataset_id, mrl_subset, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, query_field=query_field, answer_field=answer_field,
                   interactions_field=interactions_field, max_seq_len=max_seq_len,
                   ignore_index=ignore_index, **kwargs)


class AutoregressiveLMIterableDataset(BaseIterableDataset):
    """
    Iterable version of AutoregressiveLMDataset for streaming datasets.

    This dataset works with HuggingFace streaming datasets and is suitable
    for large-scale datasets that don't fit in memory.
    """

    def __init__(
            self,
            hf_iterable_dataset: HfIterableDataset,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            hf_field: str = 'text',
            **kwargs
    ):
        super().__init__(hf_iterable_dataset, tokenizer, max_seq_len, hf_field, **kwargs)

    def __iter__(self):
        """
        Iterate through the dataset, yielding tokenized examples.

        Yields:
            dict with keys 'input_ids', 'attention_mask', and 'targets'
        """
        tokenized_iterator = super().__iter__()

        for tokenized_inputs in tokenized_iterator:
            input_ids = tokenized_inputs['input_ids'][0]
            attention_mask = tokenized_inputs['attention_mask'][0]
            targets = input_ids.clone()

            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'targets': targets
            }

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            subset: str = None,
            split: str = 'train',
            target_field: str = 'text',
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            **kwargs
    ):
        """
        Load streaming dataset from HuggingFace Hub for autoregressive language modeling.

        Args:
            dataset_id (str): Hub dataset repository name
            subset (str): Dataset subset (default: None)
            split (str): Dataset split (default: "train")
            target_field (str): Name of dataset field used for training (default: "text")
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): HuggingFace Tokenizer (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer
            **kwargs: Additional args for the dataset class

        Returns:
            AutoregressiveLMIterableDataset instance
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, \
            "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_dataset = load_dataset(
            dataset_id,
            name=subset,
            split=split,
            streaming=True,
            **load_kwargs
        )

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)
