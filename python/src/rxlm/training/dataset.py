import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from datasets import Dataset as HfDataset, load_dataset, concatenate_datasets, IterableDataset as HfIterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .tokenizer import load_tokenizer_from_hf_hub
import torch.distributed as dist

from typing import Union, TypedDict, Optional, TypeAlias, Any, Literal


class BaseDataset(Dataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            hf_field: str = 'text',
            cache_tokenized: bool = False,
            cache_remove_text: bool = True,
            tokenize_in_background: bool = False,
            batch_size: int = 1,
            *args,
            **kwargs
    ):
        super(BaseDataset, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = texts
        self.hf_field = hf_field
        self.is_pre_tokenized = False
        self.cache_tokenized = cache_tokenized
        self.cache_remove_text = cache_remove_text
        self.inputs = []
        self.is_txt_list = isinstance(self.texts, list)
        self.tokenize_in_background = tokenize_in_background
        self.bg_next = []
        self.bg_queue = None
        self.batch_size = batch_size
        self.last_idx = 0
        if tokenize_in_background:
            for i in range(self.batch_size):
                self.bg_next.append(self.get_tokenized_text(i))
            self.last_idx = self.batch_size - 1

    def __len__(self):
        return len(self.texts if not self.is_pre_tokenized else self.inputs)

    def get_tokenized_text(self, idx: int, txt: str = None):
        if self.is_pre_tokenized:
            return self.inputs[idx]
        elif self.tokenize_in_background:
            if idx == self.last_idx - self.batch_size:
                if self.bg_queue is not None:
                    self.bg_next = self.bg_queue
                    self.bg_queue = None
                # TODO: schedule tokenizing a batch in background
            elif idx == self.last_idx:
                item = self.bg_next[idx]
                self.bg_next = []
                return item

            if idx <= self.last_idx:
                if self.bg_queue is not None:
                    self.bg_next = self.bg_queue
                    self.bg_queue = None

                new_idx = idx - (self.last_idx - self.batch_size)
                if new_idx in self.bg_next:
                    return self.bg_next[new_idx]
                else:
                    if self.is_txt_list:
                        text = self.texts[idx]
                    else:
                        text = self.texts[idx][self.hf_field]

                    inputs = self.tokenizer(
                        text,
                        max_length=self.max_seq_len,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                        return_attention_mask=True
                    )
                    if not (inputs['input_ids'][0] < self.tokenizer.vocab_size).all():
                        inputs['input_ids'][0][
                            (inputs['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
                    if not (inputs['input_ids'][0] >= 0).all():
                        inputs['input_ids'][0][inputs['input_ids'][0] < 0] = self.tokenizer.unk_token_id

                    return inputs
        else:
            if txt is not None:
                text = txt
            elif self.is_txt_list:
                text = self.texts[idx]
            else:
                text = self.texts[idx][self.hf_field]

            inputs = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True
            )
            if not (inputs['input_ids'][0] < self.tokenizer.vocab_size).all():
                inputs['input_ids'][0][
                    (inputs['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
            if not (inputs['input_ids'][0] >= 0).all():
                inputs['input_ids'][0][inputs['input_ids'][0] < 0] = self.tokenizer.unk_token_id

            if self.cache_tokenized:
                self.inputs.append(inputs)
                if len(self.inputs) == len(self.texts):
                    self.is_pre_tokenized = True
                    if self.cache_remove_text:
                        del self.texts
                        self.texts = None

            return inputs

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "BaseDataset":
        split_point = int(len(self.texts) * ((1 - size) if not from_start else size))
        if not isinstance(self.texts, list):
            subset = self.texts.select(range(split_point, len(self.texts)) if not from_start else range(split_point))
            self.texts = self.texts.select(
                range(split_point) if not from_start else range(split_point, len(self.texts)))
        else:
            subset = self.texts[split_point:-1] if not from_start else self.texts[0:split_point]
            self.texts = self.texts[0:split_point] if not from_start else self.texts[split_point:-1]
        return self.__class__(subset, self.tokenizer, max_seq_len=self.max_seq_len, hf_field=self.hf_field, **kwargs)

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000, map_hf_ds_to_list: bool = True):
        """
        Pre-tokenizes all the items in the dataset, for faster training. Training with pre-tokenized
        dataset could be even 2x faster.

        !! This method has extremely high memory usage, when used with HuggingFace datasets,
        because of converting it to list. Additionally, for the most optimal performance,
        pre-tokenized items are in reversed order - it shouldn't matter for training, as
        items are shuffled then by DataLoader, but you should keep that in mind in case
        of reproducibility.

        :param(bool) verbose:
        :param(int) log_interval: Interval of verbose logs
        """
        if not self.is_pre_tokenized:
            num_texts = len(self.texts)
            txts = self.texts if self.is_txt_list else self.texts.to_list()
            del self.texts
            self.texts = None
            for index in range(num_texts):
                item = txts.pop() if self.is_txt_list else txts.pop()[self.hf_field]
                self.inputs.append(self.get_tokenized_text(index, txt=item))
                if verbose and index % log_interval == 0:
                    print(f'Processed {index + 1}/{num_texts}')
            self.is_pre_tokenized = True

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
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_id (str): Hub dataset repository name
            subset (str): Dataset subset
            split (str): Dataset split (default: "train")
            target_field (str): Name of dataset field used for training (default: "text")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_dataset = load_dataset(dataset_id, subset, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)

    @classmethod
    def concat_from_hf_hub(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
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
        Load and concatenate multiple datasets from HuggingFace Hub and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            target_field (str): Name of dataset field used for training (default: "text")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]
        hf_dataset = concatenate_datasets(hf_datasets)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)

    @classmethod
    def concat_from_hf_hub_with_subset(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            split: str = 'train',
            target_field: str = 'text',
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            valid_size: Union[float, tuple[int]] = 0.1,
            **kwargs
    ):
        """
        Load and concatenate multiple datasets from HuggingFace Hub, create validation split and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            target_field (str): Name of dataset field used for training (default: "text")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            valid_size (float): Size of validation dataset  (default: 0.1)
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]

        if isinstance(valid_size, float):
            hf_ds_dicts = [dataset.train_test_split(test_size=valid_size) for dataset in hf_datasets]
        else:
            hf_ds_dicts = [dataset.train_test_split(test_size=valid_size[i]) for i, dataset in enumerate(hf_datasets)]

        hf_dataset = concatenate_datasets([ds_dict['train'] for ds_dict in hf_ds_dicts])
        hf_valid_dataset = concatenate_datasets([ds_dict['test'] for ds_dict in hf_ds_dicts])

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs), cls(
            hf_valid_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)


class JointLMDataset(BaseDataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: PreTrainedTokenizer,
            max_seq_len: int = 1024,
            mask_prob: float = 0.15,
            hf_field: str = 'text',
            *args,
            **kwargs
    ):
        super(JointLMDataset, self).__init__(texts, tokenizer, max_seq_len, hf_field, *args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        inputs = self.get_tokenized_text(idx)
        encoder_input_ids = inputs['input_ids'][0]
        if self.is_pre_tokenized:
            encoder_input_ids = encoder_input_ids.clone()
        attention_mask = inputs['attention_mask'][0]

        decoder_input_ids = encoder_input_ids.clone()

        encoder_labels = encoder_input_ids.clone()
        decoder_targets = encoder_input_ids.clone()

        # Create masked indices
        masked_indices = torch.bernoulli(
            torch.full(encoder_labels.shape, self.mask_prob)
        ).bool() & attention_mask.bool()

        # Apply mask
        encoder_labels[~masked_indices] = -100
        encoder_input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'decoder': {
                'input_ids': decoder_input_ids,
                'targets': decoder_targets,
            },
            'encoder': {
                'input_ids': encoder_input_ids,
                'labels': encoder_labels,
            },
            'attention_mask': attention_mask,
        }


class BaseInteractionDataset(Dataset):
    def __init__(
            self,
            interactions: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            query_field: str = 'query',
            answer_field: str = 'answer',
            cache_tokenized: bool = False,
            cache_remove_text: bool = True,
            tokenize_in_background: bool = False,
            batch_size: int = 1,
            *args,
            **kwargs
    ):
        super(BaseInteractionDataset, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.interactions = interactions
        self.query_field = query_field
        self.answer_field = answer_field
        self.is_pre_tokenized = False
        self.cache_tokenized = cache_tokenized
        self.cache_remove_text = cache_remove_text
        self.inputs = []
        self.is_list = isinstance(self.interactions, list)
        self.tokenize_in_background = tokenize_in_background
        self.bg_next = []
        self.bg_queue = None
        self.batch_size = batch_size
        self.last_idx = 0
        if tokenize_in_background:
            for i in range(self.batch_size):
                self.bg_next.append(self.get_tokenized_text(i))
            self.last_idx = self.batch_size - 1

    def __len__(self):
        return len(self.interactions if not self.is_pre_tokenized else self.inputs)

    def get_tokenized_text(self, idx: int, inter: dict = None):
        if self.is_pre_tokenized:
            return self.inputs[idx]
        elif self.tokenize_in_background:
            if idx == self.last_idx - self.batch_size:
                if self.bg_queue is not None:
                    self.bg_next = self.bg_queue
                    self.bg_queue = None
                # TODO: schedule tokenizing a batch in background
            elif idx == self.last_idx:
                item = self.bg_next[idx]
                self.bg_next = []
                return item

            if idx <= self.last_idx:
                if self.bg_queue is not None:
                    self.bg_next = self.bg_queue
                    self.bg_queue = None

                new_idx = idx - (self.last_idx - self.batch_size)
                if new_idx in self.bg_next:
                    return self.bg_next[new_idx]
                else:
                    interaction = self.interactions[idx]
                    query = interaction[self.query_field]
                    answer = interaction[self.answer_field]

                    inputs = self.tokenizer(
                        query,
                        answer,
                        max_length=self.max_seq_len,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                        return_attention_mask=True
                    )
                    if not (inputs['input_ids'][0] < self.tokenizer.vocab_size).all():
                        inputs['input_ids'][0][
                            (inputs['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
                    if not (inputs['input_ids'][0] >= 0).all():
                        inputs['input_ids'][0][inputs['input_ids'][0] < 0] = self.tokenizer.unk_token_id

                    return inputs
        else:
            if inter is not None:
                interaction = inter
            else:
                interaction = self.interactions[idx]
            query = interaction[self.query_field]
            answer = interaction[self.answer_field]

            inputs = self.tokenizer(
                query,
                answer,
                max_length=self.max_seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True
            )
            if not (inputs['input_ids'][0] < self.tokenizer.vocab_size).all():
                inputs['input_ids'][0][
                    (inputs['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
            if not (inputs['input_ids'][0] >= 0).all():
                inputs['input_ids'][0][inputs['input_ids'][0] < 0] = self.tokenizer.unk_token_id

            if self.cache_tokenized:
                self.inputs.append(inputs)
                if len(self.inputs) == len(self.interactions):
                    self.is_pre_tokenized = True
                    if self.cache_remove_text:
                        del self.interactions
                        self.interactions = None

            return inputs

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "BaseInteractionDataset":
        split_point = int(len(self.interactions) * ((1 - size) if not from_start else size))
        if not isinstance(self.interactions, list):
            subset = self.interactions.select(
                range(split_point, len(self.interactions)) if not from_start else range(split_point))
            self.interactions = self.interactions.select(
                range(split_point) if not from_start else range(split_point, len(self.interactions)))
        else:
            subset = self.interactions[split_point:-1] if not from_start else self.interactions[0:split_point]
            self.interactions = self.interactions[0:split_point] if not from_start else self.interactions[
                                                                                        split_point:-1]
        return self.__class__(subset, self.tokenizer, max_seq_len=self.max_seq_len, query_field=self.query_field,
                              answer_field=self.answer_field, **kwargs)

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000):
        """
        Pre-tokenizes all the items in the dataset, for faster training. Training with pre-tokenized
        dataset could be even 2x faster.

        !! This method has extremely high memory usage, when used with HuggingFace datasets,
        because of converting it to list. Additionally, for the most optimal performance,
        pre-tokenized items are in reversed order - it shouldn't matter for training, as
        items are shuffled then by DataLoader, but you should keep that in mind in case
        of reproducibility.

        :param(bool) verbose:
        :param(int) log_interval: Interval of verbose logs
        """
        if not self.is_pre_tokenized:
            num_texts = len(self.interactions)
            inters = self.interactions if self.is_list else self.interactions.to_list()
            del self.interactions
            self.interactions = None
            for index in range(num_texts):
                self.inputs.append(self.get_tokenized_text(index, inter=inters.pop()))
                if verbose and index % log_interval == 0:
                    print(f'Processed {index + 1}/{num_texts}')
            self.is_pre_tokenized = True

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            subset: str = None,
            split: str = 'train',
            target_fields: tuple[str, str] = ('query', 'answer'),
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            **kwargs
    ):
        """
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_id (str): Hub dataset repository name
            subset (str): Dataset subset
            split (str): Dataset split (default: "train")
            target_fields (tuple): Name of dataset fields used for training (default: ("query", "answer"))
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_dataset = load_dataset(dataset_id, subset, split=split, **load_kwargs)

        query_field, answer_field = target_fields

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field, answer_field=answer_field,
                   **kwargs)

    @classmethod
    def concat_from_hf_hub(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            split: str = 'train',
            target_fields: tuple[str, str] = ('query', 'answer'),
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            **kwargs
    ):
        """
        Load and concatenate multiple datasets from HuggingFace Hub and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            target_fields (tuple): Name of dataset field used for training (default: ("query", "answer"))
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]
        hf_dataset = concatenate_datasets(hf_datasets)

        query_field, answer_field = target_fields

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field, answer_field=answer_field,
                   **kwargs)

    @classmethod
    def concat_from_hf_hub_with_subset(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            split: str = 'train',
            target_fields: tuple[str, str] = ('query', 'answer'),
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            valid_size: float = 0.1,
            **kwargs
    ):
        """
        Load and concatenate multiple datasets from HuggingFace Hub, create validation split and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            target_fields (tuple[str, str]): Name of dataset field used for training (default: "text")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            valid_size (float): Size of validation dataset  (default: 0.1)
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if load_kwargs is None:
            load_kwargs = {}

        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]

        hf_ds_dicts = [dataset.train_test_split(test_size=valid_size) for dataset in hf_datasets]

        hf_dataset = concatenate_datasets([ds_dict['train'] for ds_dict in hf_ds_dicts])
        hf_valid_dataset = concatenate_datasets([ds_dict['test'] for ds_dict in hf_ds_dicts])

        query_field, answer_field = target_fields

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field, answer_field=answer_field,
                   **kwargs), cls(hf_valid_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field,
                                  answer_field=answer_field, **kwargs)


class JointSftDataset(BaseInteractionDataset):
    def __init__(
            self,
            interactions: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            query_field: str = 'query',
            answer_field: str = 'answer',
            cache_tokenized: bool = False,
            cache_remove_text: bool = True,
            tokenize_in_background: bool = False,
            batch_size: int = 1,
            mask_prob: float = 0.15,
            ignore_index: int = -100,
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            *args,
            **kwargs
    ):
        super(JointSftDataset, self).__init__(
            interactions,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            query_field=query_field,
            answer_field=answer_field,
            cache_tokenized=cache_tokenized,
            cache_remove_text=cache_remove_text,
            tokenize_in_background=tokenize_in_background,
            batch_size=batch_size,
            *args,
            **kwargs
        )
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index
        self.query_token = query_token
        self.answer_token = answer_token
        self.bos_token = bos_token
        self.eos_token = eos_token

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

    def get_tokenized_text(self, idx: int, inter: dict = None):
        inputs = super().get_tokenized_text(idx, inter=inter)

        decoder_targets = self._create_masked_labels(inputs['input_ids'][0])

        return {
            **inputs,
            'decoder_targets': decoder_targets,
        }

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        inputs = self.get_tokenized_text(idx)

        encoder_input_ids = inputs['input_ids'][0]
        if self.is_pre_tokenized:
            encoder_input_ids = encoder_input_ids.clone()
        attention_mask = inputs['attention_mask'][0]

        decoder_input_ids = encoder_input_ids.clone()

        encoder_labels = encoder_input_ids.clone()
        decoder_targets = inputs['decoder_targets']

        # Create masked indices
        masked_indices = torch.bernoulli(
            torch.full(encoder_labels.shape, self.mask_prob)
        ).bool() & attention_mask.bool()

        # Apply mask
        encoder_labels[~masked_indices] = -100
        encoder_input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'decoder': {
                'input_ids': decoder_input_ids,
                'targets': decoder_targets,
            },
            'encoder': {
                'input_ids': encoder_input_ids,
                'labels': encoder_labels,
            },
            'attention_mask': attention_mask,
        }


class DecoderSftDataset(BaseInteractionDataset):
    def __init__(
            self,
            interactions: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            query_field: str = 'query',
            answer_field: str = 'answer',
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            ignore_index: int = -100,
            **kwargs,
    ):
        super(DecoderSftDataset, self).__init__(
            interactions,
            tokenizer,
            max_seq_len=max_seq_len,
            query_field=query_field,
            answer_field=answer_field,
            **kwargs
        )
        self.query_token = query_token
        self.answer_token = answer_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.ignore_index = ignore_index

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inputs = self.get_tokenized_text(idx)

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        targets = self._create_masked_labels(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }


class EncoderSftDataset(BaseInteractionDataset):
    def __init__(
            self,
            interactions: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            query_field: str = 'query',
            answer_field: str = 'answer',
            cache_tokenized: bool = False,
            cache_remove_text: bool = True,
            tokenize_in_background: bool = False,
            batch_size: int = 1,
            mask_prob: float = 0.15,
            *args,
            **kwargs
    ):
        super(EncoderSftDataset, self).__init__(
            interactions,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            query_field=query_field,
            answer_field=answer_field,
            cache_tokenized=cache_tokenized,
            cache_remove_text=cache_remove_text,
            tokenize_in_background=tokenize_in_background,
            batch_size=batch_size,
            *args,
            **kwargs
        )
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

ItemFields: TypeAlias = Literal['input_ids', 'attention_mask']
MrlDataItem: TypeAlias = dict[str, Union[dict[ItemFields, torch.Tensor], list[dict[str, dict[ItemFields, torch.Tensor]]]]]


class MrlCurriculumDataset(Dataset):
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
            **kwargs,
    ):
        super(MrlCurriculumDataset, self).__init__(**kwargs)
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
        self.is_pre_tokenized = False
        self.is_list = isinstance(self.episodes, list)
        self.inputs = []

    def _tokenize_manual_interaction(self, query: str, answer: str) -> dict[str, dict[str, torch.Tensor]]:
        # Manually construct query: [BOS][Q]query
        query_text = f"{self.bos_token}{self.query_token}{query}"
        query_enc = self.tokenizer(
            query_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Critical: We control all tokens
        )

        if not (query_enc['input_ids'][0] < self.tokenizer.vocab_size).all():
            query_enc['input_ids'][0][
                (query_enc['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
        if not (query_enc['input_ids'][0] >= 0).all():
            query_enc['input_ids'][0][query_enc['input_ids'][0] < 0] = self.tokenizer.unk_token_id

        # Manually construct answer: [A]answer[EOS]
        answer_text = f"{self.answer_token}{answer}{self.eos_token}"
        answer_enc = self.tokenizer(
            answer_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Critical: We control all tokens
        )

        if not (answer_enc['input_ids'][0] < self.tokenizer.vocab_size).all():
            answer_enc['input_ids'][0][
                (answer_enc['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
        if not (answer_enc['input_ids'][0] >= 0).all():
            answer_enc['input_ids'][0][answer_enc['input_ids'][0] < 0] = self.tokenizer.unk_token_id

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

    def get_tokenized_item(self, idx: int, episode: dict = None) -> MrlDataItem:
        if self.is_pre_tokenized:
            return self.inputs[idx]
        else:
            item = self.episodes[idx] if episode is None else episode
            query = item[self.query_field]
            answer = item[self.answer_field]
            interactions = item[self.interactions_field]

            initial = self._tokenize_manual_interaction(query, answer)
            follow_ups = [self._tokenize_manual_interaction(inter[self.query_field], inter[self.answer_field]) for inter in interactions]

            return {
                **initial,
                'interactions': follow_ups,
            }

    def __getitem__(self, idx: int) -> MrlDataItem:
        return self.get_tokenized_item(idx)

    def __len__(self) -> int:
        return len(self.inputs if self.is_pre_tokenized else self.episodes)

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "MRlCurriculumDataset":
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
            max_seq_len: int = 1024,
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
            **kwargs: Additional args for RxNN Dataset class
        """
        if load_kwargs is None:
            load_kwargs = {}

        hf_dataset = load_dataset(dataset_id, mrl_subset, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, query_field=query_field, answer_field=answer_field,
                   interactions_field=interactions_field, max_seq_len=max_seq_len, **kwargs)

    @staticmethod
    def collate_mrl_batch(batch: list[MrlDataItem]) -> MrlDataItem:
        """Collate function for MRL curriculum dataset with nested interactions"""

        def collate_interaction_batch(interaction_batch: Union[list[dict[str, dict[str, torch.Tensor]]], tuple[Any]]) -> \
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

        batch_interactions = [x['interactions'] for x in batch]
        transposed_interactions = list(zip(*batch_interactions))

        return {
            **collate_interaction_batch(batch),  # Collate initial query and answer
            'interactions': [
                collate_interaction_batch(step_batch) for step_batch in transposed_interactions
            ]
        }


class MrlDatasetItem(TypedDict):
    steps: int
    is_long_range: bool
    dataset: MrlCurriculumDataset
    eval_dataset: Optional[MrlCurriculumDataset]


class MrlDatasetLoadItem(TypedDict):
    subset_name: str
    steps: int
    is_long_range: bool


class MrlDatasets:
    def __init__(self, datasets: list[MrlDatasetItem]):
        self.datasets = datasets

    def __iter__(self):
        return iter(self.datasets)

    def __getitem__(self, idx: int) -> MrlDatasetItem:
        return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)

    def __call__(self, steps: int, is_long_range: bool = False):
        for dataset in self.datasets:
            if dataset['steps'] == steps and dataset['is_long_range'] == is_long_range:
                return dataset
        return None

    @property
    def is_pre_tokenized(self) -> bool:
        train_tokenized = all(item['dataset'].is_pre_tokenized for item in self.datasets)
        eval_tokenized = all(
            item['eval_dataset'].is_pre_tokenized for item in self.datasets if item['eval_dataset'] is not None)
        return train_tokenized and eval_tokenized

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000, keep_order: bool = False):
        """
        Pre-tokenizes all the inner datasets

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
            for item in self.datasets:
                item['dataset'].pre_tokenize(verbose, log_interval, keep_order)
                if item['eval_dataset'] is not None:
                    item['eval_dataset'].pre_tokenize(verbose, log_interval, keep_order)

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            mrl_curriculum_steps: Union[list[MrlDatasetLoadItem], tuple[MrlDatasetLoadItem]],
            split: str = 'train',
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            load_kwargs: dict = None,
            mrl_ds_kwargs: dict = None,
            eval_split: str = None,
            max_seq_len: int = 256,
    ):
        """
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_id (str): Hub dataset repository name
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer
            mrl_curriculum_steps (list[MrlDatasetLoadItem]): MRL Curriculum steps configuration
            split (str): Dataset split (default: "train")
            query_field (str): Query field (default: "query")
            answer_field (str): Answer field (default: "answer")
            interactions_field (str): Interactions field (default: "interactions")
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            mrl_ds_kwargs (dict): Additional args for RxNN MrlCurriculumDataset class
            eval_split (str): Load also evaluation/validation split (default: None)
            max_seq_len (int): Maximum sequence length (default: 256)
        """
        if load_kwargs is None:
            load_kwargs = {}
        if mrl_ds_kwargs is None:
            mrl_ds_kwargs = {}

        def load_subset(subset_name: str, load_split: str):
            return MrlCurriculumDataset.from_hf_hub(
                dataset_id,
                subset_name,
                tokenizer=tokenizer,
                query_field=query_field,
                answer_field=answer_field,
                interactions_field=interactions_field,
                split=load_split,
                load_kwargs=load_kwargs,
                max_seq_len=max_seq_len,
                **mrl_ds_kwargs,
            )

        def dataset_item(item: MrlDatasetLoadItem) -> MrlDatasetItem:
            return {
                'steps': item['steps'],
                'is_long_range': item['is_long_range'],
                'dataset': load_subset(item['subset_name'], split),
                'eval_dataset': load_subset(item['subset_name'], eval_split) if eval_split is not None else None,
            }

        mrl_datasets = [dataset_item(item) for item in mrl_curriculum_steps]

        return cls(mrl_datasets)


class BaseIterableDataset(IterableDataset):
    def __init__(
            self,
            hf_iterable_dataset: HfIterableDataset,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            hf_field: str = 'text',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_dataset = hf_iterable_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.hf_field = hf_field

    def tokenize_text(self, text: str):
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )

        input_ids = inputs['input_ids'][0]
        vocab_size = self.tokenizer.vocab_size
        unk_token_id = self.tokenizer.unk_token_id

        input_ids[input_ids >= vocab_size] = unk_token_id
        input_ids[input_ids < 0] = unk_token_id

        return inputs

    def __iter__(self):
        for idx, example in enumerate(self.hf_dataset):
            text = example[self.hf_field]
            yield self.tokenize_text(text)

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
        assert tokenizer is not None or tokenizer_hub_id is not None, "Musisz podać `tokenizer` lub `tokenizer_hub_id`."

        if load_kwargs is None: load_kwargs = {}
        if load_tokenizer_kwargs is None: load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True, **load_kwargs)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)


class JointLMIterableDataset(BaseIterableDataset):
    def __init__(
            self,
            hf_iterable_dataset: HfIterableDataset,
            tokenizer: PreTrainedTokenizer,
            max_seq_len: int = 1024,
            mask_prob: float = 0.15,
            hf_field: str = 'text',
            **kwargs
    ):
        super().__init__(hf_iterable_dataset, tokenizer, max_seq_len, hf_field, **kwargs)
        self.mask_prob = mask_prob

    def __iter__(self):
        tokenized_iterator = super().__iter__()

        for tokenized_inputs in tokenized_iterator:
            encoder_input_ids = tokenized_inputs['input_ids'][0].clone()
            attention_mask = tokenized_inputs['attention_mask'][0]
            decoder_input_ids = encoder_input_ids.clone()
            decoder_targets = encoder_input_ids.clone()

            masked_indices = torch.bernoulli(
                torch.full(encoder_input_ids.shape, self.mask_prob)
            ).bool() & attention_mask.bool()

            encoder_labels = encoder_input_ids.clone()
            encoder_labels[~masked_indices] = -100
            encoder_input_ids[masked_indices] = self.tokenizer.mask_token_id

            yield {
                'decoder': {
                    'input_ids': decoder_input_ids,
                    'targets': decoder_targets,
                },
                'encoder': {
                    'input_ids': encoder_input_ids,
                    'labels': encoder_labels,
                },
                'attention_mask': attention_mask,
            }


class SmatDataset(MrlCurriculumDataset):
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
            use_system_prompt: bool = False,
            system_field: str = 'system',
            system_prompt_title: str = 'SYSTEM INSTRUCTIONS',
            **kwargs,
    ):
        super(SmatDataset, self).__init__(
            episodes,
            tokenizer,
            max_seq_len,
            query_field,
            answer_field,
            interactions_field,
            query_token,
            answer_token,
            bos_token,
            eos_token,
            **kwargs
        )
        self.system_field = system_field
        self.use_system_prompt = use_system_prompt
        self.system_prompt_title = system_prompt_title

    def _tokenize_system_prompt(self, system: str) -> dict[str, dict[str, torch.Tensor]]:
        # Manually construct query: [BOS][Q]query
        system_text = f"{self.bos_token}{self.query_token}{self.system_prompt_title}{self.answer_token}{system}{self.eos_token}"
        system_enc = self.tokenizer(
            system_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Critical: We control all tokens
        )

        return {
            'system': {
                'input_ids': system_enc['input_ids'][0],
                'attention_mask': system_enc['attention_mask'][0],
            },

        }

    def get_tokenized_item(self, idx: int, episode: dict = None) -> MrlDataItem:
        if self.is_pre_tokenized:
            return self.inputs[idx]
        else:
            item = self.episodes[idx] if episode is None else episode
            query = item[self.query_field]
            answer = item[self.answer_field]
            interactions = item[self.interactions_field]

            initial = self._tokenize_manual_interaction(query, answer)
            follow_ups = [self._tokenize_manual_interaction(inter[self.query_field], inter[self.answer_field]) for inter
                          in interactions]

            if self.use_system_prompt:
                system_prompt = item[self.system_field]
                system_enc = self._tokenize_system_prompt(system_prompt) if system_prompt is not None else None
                return {
                    **initial,
                    'system': system_enc,
                    'interactions': follow_ups,
                }
            else:
                return {
                    **initial,
                    'interactions': follow_ups,
                }

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            subset: str = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            split: str = 'train',
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            load_kwargs: dict = None,
            max_seq_len: int = 1024,
            **kwargs
    ):
        """
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_id (str): Hub dataset repository name
            subset (str): Dataset subset
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer
            split (str): Dataset split (default: "train")
            query_field (str): Query field (default: "query")
            answer_field (str): Answer field (default: "answer")
            interactions_field (str): Interactions field (default: "interactions")
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            max_seq_len (int): Maximum sequence length (default: 1024)
            **kwargs: Additional args for RxNN Dataset class
        """
        if load_kwargs is None:
            load_kwargs = {}

        hf_dataset = load_dataset(dataset_id, subset, split=split, **load_kwargs) if subset is not None else load_dataset(dataset_id, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, query_field=query_field, answer_field=answer_field,
                   interactions_field=interactions_field, max_seq_len=max_seq_len, **kwargs)

    @classmethod
    def concat_from_hf_hub(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            split: str = 'train',
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            **kwargs
    ):
        """
        Load and concatenate multiple datasets from HuggingFace Hub, create validation split and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            query_field (str): Query field (default: "query")
            answer_field (str): Answer field (default: "answer")
            interactions_field (str): Interactions field (default: "interactions")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            **kwargs: Additional args for RxNN Dataset class
        """
        if load_kwargs is None:
            load_kwargs = {}

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]

        hf_dataset = concatenate_datasets(hf_datasets)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field, answer_field=answer_field, interactions_field=interactions_field, **kwargs)


    @classmethod
    def concat_from_hf_hub_with_subset(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            split: str = 'train',
            query_field: str = 'query',
            answer_field: str = 'answer',
            interactions_field: str = 'interactions',
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            valid_size: Union[float, tuple[int]] = 0.1,
            **kwargs
    ):
        """
        Load and concatenate multiple datasets from HuggingFace Hub, create validation split and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            query_field (str): Query field (default: "query")
            answer_field (str): Answer field (default: "answer")
            interactions_field (str): Interactions field (default: "interactions")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            valid_size (float): Size of validation dataset  (default: 0.1)
            **kwargs: Additional args for RxNN Dataset class
        """
        if load_kwargs is None:
            load_kwargs = {}

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]

        if isinstance(valid_size, float):
            hf_ds_dicts = [dataset.train_test_split(test_size=valid_size) for dataset in hf_datasets]
        else:
            hf_ds_dicts = [dataset.train_test_split(test_size=valid_size[i]) for i, dataset in enumerate(hf_datasets)]

        hf_dataset = concatenate_datasets([ds_dict['train'] for ds_dict in hf_ds_dicts])
        hf_valid_dataset = concatenate_datasets([ds_dict['test'] for ds_dict in hf_ds_dicts])

        return (
            cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field, answer_field=answer_field, interactions_field=interactions_field, **kwargs),
            cls(hf_valid_dataset, tokenizer, max_seq_len=max_seq_len, query_field=query_field, answer_field=answer_field, interactions_field=interactions_field, **kwargs)
        )

    @staticmethod
    def collate_smat_batch(batch: list[MrlDataItem]) -> MrlDataItem:
        """Collate function for MRL curriculum dataset with nested interactions"""

        def collate_interaction_batch(interaction_batch: Union[list[dict[str, dict[str, torch.Tensor]]], tuple[Any]]) -> \
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

        batch_interactions = [x['interactions'] for x in batch]
        transposed_interactions = list(zip(*batch_interactions))

        return {
            **collate_interaction_batch(batch),  # Collate initial query and answer
            'interactions': [
                collate_interaction_batch(step_batch) for step_batch in transposed_interactions
            ]
        }

    @staticmethod
    def collate_smat_batch_with_system(batch: list[MrlDataItem]) -> MrlDataItem:
        """Collate function for MRL curriculum dataset with nested interactions"""

        def collate_interaction_batch(interaction_batch: Union[list[dict[str, dict[str, torch.Tensor]]], tuple[Any]]) -> \
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

        def collate_system_prompt(interaction_batch: Union[list[dict[str, dict[str, torch.Tensor]]], tuple[Any]]) -> \
                dict[str, dict[ItemFields, torch.Tensor]]:
            """Helper to collate a batch of interactions"""

            empty_ids = torch.zeros_like(interaction_batch[0]['query']['input_ids'], dtype=torch.long)
            empty_mask = torch.zeros_like(interaction_batch[0]['query']['attention_mask'], dtype=torch.long)

            def get_input_ids(x: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
                return x['system']['input_ids'] if x['system'] is not None else empty_ids.clone()

            def get_attention_mask(x: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
                return x['system']['attention_mask'] if x['system'] is not None else empty_mask.clone()

            return {
                'system': {
                    'input_ids': torch.stack([get_input_ids(x) for x in interaction_batch]),
                    'attention_mask': torch.stack([get_attention_mask(x) for x in interaction_batch]),
                },
            }

        batch_interactions = [x['interactions'] for x in batch]
        transposed_interactions = list(zip(*batch_interactions))

        return {
            **collate_interaction_batch(batch),  # Collate initial query and answer
            **collate_system_prompt(batch),
            'interactions': [
                collate_interaction_batch(step_batch) for step_batch in transposed_interactions
            ]
        }


class HybridReasoningSftDataset(Dataset):
    """
    Dataset for Interaction SFT with hybrid reasoning and agentic tools support.

    Supports the new RxT-Beta Interaction Template with special tokens:
    - [Q] - query block
    - [A] - answer block
    - [T] - thinking/reasoning block
    - [C] - agentic tool call
    - [U] - agentic tool use
    - [I] - internal instruction

    The template structure is:
    - Fast answer: [I] internal [Q] query [A] answer [C] tool_call
    - Extended thinking: [I] internal [Q] query [T] think [A] answer [C] tool_call
    - Tool use: [I] internal [U] tool_use [T] think [A] answer [C] tool_call

    For training loss:
    - Masked (ignored): internal, query, tool_use tokens
    - Used for loss: think, answer, tool_call tokens
    """

    def __init__(
            self,
            interactions: Union[list[dict], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            # Field names in dataset
            query_field: str = 'query',
            answer_field: str = 'answer',
            think_field: str = 'think',
            tool_call_field: str = 'tool_call',
            tool_use_field: str = 'tool_use',
            internal_field: str = 'internal',
            # Special tokens (text versions for manual tokenization)
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            think_token: str = '[T]',
            tool_call_token: str = '[C]',
            tool_use_token: str = '[U]',
            internal_token: str = '[I]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            # Other options
            ignore_index: int = -100,
            mask_prob: float = 0.15,
            cache_tokenized: bool = False,
            cache_remove_text: bool = True,
            **kwargs,
    ):
        super(HybridReasoningSftDataset, self).__init__(**kwargs)
        self.interactions = interactions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Field names
        self.query_field = query_field
        self.answer_field = answer_field
        self.think_field = think_field
        self.tool_call_field = tool_call_field
        self.tool_use_field = tool_use_field
        self.internal_field = internal_field

        # Special tokens
        self.query_token = query_token
        self.answer_token = answer_token
        self.think_token = think_token
        self.tool_call_token = tool_call_token
        self.tool_use_token = tool_use_token
        self.internal_token = internal_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.ignore_index = ignore_index
        self.mask_prob = mask_prob
        self.cache_tokenized = cache_tokenized
        self.cache_remove_text = cache_remove_text

        self.is_pre_tokenized = False
        self.is_list = isinstance(self.interactions, list)
        self.inputs = []

    def __len__(self):
        return len(self.inputs if self.is_pre_tokenized else self.interactions)

    def _build_interaction_text(self, item: dict) -> str:
        """
        Builds the full interaction text from components.

        Template structure:
        [BOS][I]internal[Q|U]query_or_tool_use[T]think[A]answer[C]tool_call[EOS]

        All components except [BOS] and [EOS] are optional.
        """
        parts = [self.bos_token]

        # Internal instruction (optional, at the start)
        internal = item.get(self.internal_field)
        if internal:
            parts.append(f"{self.internal_token}{internal}")

        # Either query or tool_use (mutually exclusive)
        tool_use = item.get(self.tool_use_field)
        query = item.get(self.query_field)

        if tool_use:
            parts.append(f"{self.tool_use_token}{tool_use}")
        elif query:
            parts.append(f"{self.query_token}{query}")

        # Think block (optional)
        think = item.get(self.think_field)
        if think:
            parts.append(f"{self.think_token}{think}")

        # Answer block (optional - might be skipped for direct tool call)
        answer = item.get(self.answer_field)
        if answer:
            parts.append(f"{self.answer_token}{answer}")

        # Tool call (optional, at the end)
        tool_call = item.get(self.tool_call_field)
        if tool_call:
            parts.append(f"{self.tool_call_token}{tool_call}")

        parts.append(self.eos_token)

        return ''.join(parts)

    def _create_masked_labels(self, input_ids: torch.Tensor, item: dict) -> torch.Tensor:
        """
        Creates masked labels for training.

        Masked (ignore_index): internal, query, tool_use tokens
        Used for loss: think, answer, tool_call tokens
        """
        labels = input_ids.clone()

        # Get token IDs
        query_token_id = self.tokenizer.convert_tokens_to_ids(self.query_token)
        answer_token_id = self.tokenizer.convert_tokens_to_ids(self.answer_token)
        think_token_id = self.tokenizer.convert_tokens_to_ids(self.think_token)
        tool_call_token_id = self.tokenizer.convert_tokens_to_ids(self.tool_call_token)
        tool_use_token_id = self.tokenizer.convert_tokens_to_ids(self.tool_use_token)
        internal_token_id = self.tokenizer.convert_tokens_to_ids(self.internal_token)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

        # Tokens that start sections to be masked (input sections)
        mask_start_tokens = {query_token_id, tool_use_token_id, internal_token_id}
        # Tokens that start sections for training (output sections)
        train_start_tokens = {think_token_id, answer_token_id, tool_call_token_id}
        # End tokens
        end_tokens = {eos_token_id}

        in_train_section = False
        for i in range(len(input_ids)):
            token = input_ids[i].item()

            if token in train_start_tokens:
                in_train_section = True
                # Mask the special token itself
                labels[i] = self.ignore_index
            elif token in mask_start_tokens:
                in_train_section = False
                labels[i] = self.ignore_index
            elif token in end_tokens:
                in_train_section = False
                labels[i] = self.ignore_index
            elif not in_train_section:
                labels[i] = self.ignore_index

        return labels

    def get_tokenized_item(self, idx: int, item: dict = None) -> dict:
        if self.is_pre_tokenized:
            return self.inputs[idx]

        if item is None:
            item = self.interactions[idx]

        # Build the full interaction text
        text = self._build_interaction_text(item)

        # Tokenize with skip_special_tokens=False to preserve our control tokens
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            add_special_tokens=False,  # We control all tokens manually
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        # Fix any out-of-vocab tokens
        if not (input_ids < self.tokenizer.vocab_size).all():
            input_ids[input_ids >= self.tokenizer.vocab_size] = self.tokenizer.unk_token_id
        if not (input_ids >= 0).all():
            input_ids[input_ids < 0] = self.tokenizer.unk_token_id

        # Create masked labels for decoder
        decoder_targets = self._create_masked_labels(input_ids, item)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_targets': decoder_targets,
        }

        if self.cache_tokenized:
            self.inputs.append(result)
            if len(self.inputs) == len(self.interactions):
                self.is_pre_tokenized = True
                if self.cache_remove_text:
                    del self.interactions
                    self.interactions = None

        return result

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        inputs = self.get_tokenized_item(idx)

        encoder_input_ids = inputs['input_ids']
        if self.is_pre_tokenized:
            encoder_input_ids = encoder_input_ids.clone()
        attention_mask = inputs['attention_mask']

        decoder_input_ids = encoder_input_ids.clone()
        decoder_targets = inputs['decoder_targets']

        encoder_labels = encoder_input_ids.clone()

        # Create masked indices for encoder MLM
        masked_indices = torch.bernoulli(
            torch.full(encoder_labels.shape, self.mask_prob)
        ).bool() & attention_mask.bool()

        # Apply mask
        encoder_labels[~masked_indices] = self.ignore_index
        encoder_input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'decoder': {
                'input_ids': decoder_input_ids,
                'targets': decoder_targets,
            },
            'encoder': {
                'input_ids': encoder_input_ids,
                'labels': encoder_labels,
            },
            'attention_mask': attention_mask,
        }

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000):
        """Pre-tokenizes all items for faster training."""
        if not self.is_pre_tokenized:
            num_items = len(self.interactions)
            items = self.interactions if self.is_list else self.interactions.to_list()
            del self.interactions
            self.interactions = None
            for index in range(num_items):
                self.inputs.append(self.get_tokenized_item(index, item=items.pop()))
                if verbose and index % log_interval == 0:
                    print(f'Processed {index + 1}/{num_items}')
            self.is_pre_tokenized = True

    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            subset: str = None,
            split: str = 'train',
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            **kwargs
    ):
        """Load dataset from HuggingFace Hub."""
        assert tokenizer is not None or tokenizer_hub_id is not None

        if load_kwargs is None:
            load_kwargs = {}
        if load_tokenizer_kwargs is None:
            load_tokenizer_kwargs = {}

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_dataset = load_dataset(dataset_id, subset, split=split, **load_kwargs) if subset else load_dataset(dataset_id, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, **kwargs)


class HybridReasoningSmatDataset(Dataset):
    """
    Dataset for SMAT (Supervised Memory-Aware Training) with hybrid reasoning and agentic tools.

    Extended version of SmatDataset that supports the new RxT-Beta Interaction Template.
    All interactions (including the first one) are now in a unified 'interactions' list field.

    Each interaction can have:
    - internal: Internal instruction (optional)
    - query: User query (optional if tool_use is present)
    - tool_use: Tool result (optional if query is present)
    - think: Reasoning/thinking block (optional)
    - answer: Model's answer (optional if direct tool_call)
    - tool_call: Tool call JSON (optional)

    The dataset tokenizes each interaction into query_part and answer_part:
    - query_part: [BOS][I]internal[Q|U]query_or_tool_use (masked for loss)
    - answer_part: [T]think[A]answer[C]tool_call[EOS] (used for training)
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
            # Special tokens
            query_token: str = '[Q]',
            answer_token: str = '[A]',
            think_token: str = '[T]',
            tool_call_token: str = '[C]',
            tool_use_token: str = '[U]',
            internal_token: str = '[I]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            **kwargs,
    ):
        super(HybridReasoningSmatDataset, self).__init__(**kwargs)
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

        # Special tokens
        self.query_token = query_token
        self.answer_token = answer_token
        self.think_token = think_token
        self.tool_call_token = tool_call_token
        self.tool_use_token = tool_use_token
        self.internal_token = internal_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.is_pre_tokenized = False
        self.is_list = isinstance(self.episodes, list)
        self.inputs = []

    def __len__(self):
        return len(self.inputs if self.is_pre_tokenized else self.episodes)

    def _tokenize_interaction(self, interaction: dict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Tokenizes a single interaction into query_part and answer_part.

        Query part: [BOS][I]internal[Q|U]query_or_tool_use
        Answer part: [T]think[A]answer[C]tool_call[EOS]
        """
        # Build query part
        query_parts = [self.bos_token]

        internal = interaction.get(self.internal_field)
        if internal:
            query_parts.append(f"{self.internal_token}{internal}")

        tool_use = interaction.get(self.tool_use_field)
        query = interaction.get(self.query_field)

        if tool_use:
            query_parts.append(f"{self.tool_use_token}{tool_use}")
        elif query:
            query_parts.append(f"{self.query_token}{query}")

        query_text = ''.join(query_parts)

        # Build answer part
        answer_parts = []

        think = interaction.get(self.think_field)
        if think:
            answer_parts.append(f"{self.think_token}{think}")

        answer = interaction.get(self.answer_field)
        if answer:
            answer_parts.append(f"{self.answer_token}{answer}")

        tool_call = interaction.get(self.tool_call_field)
        if tool_call:
            answer_parts.append(f"{self.tool_call_token}{tool_call}")

        answer_parts.append(self.eos_token)
        answer_text = ''.join(answer_parts)

        # Tokenize query part
        query_enc = self.tokenizer(
            query_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,
        )

        query_input_ids = query_enc['input_ids'][0]
        if not (query_input_ids < self.tokenizer.vocab_size).all():
            query_input_ids[query_input_ids >= self.tokenizer.vocab_size] = self.tokenizer.unk_token_id
        if not (query_input_ids >= 0).all():
            query_input_ids[query_input_ids < 0] = self.tokenizer.unk_token_id

        # Tokenize answer part
        answer_enc = self.tokenizer(
            answer_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,
        )

        answer_input_ids = answer_enc['input_ids'][0]
        if not (answer_input_ids < self.tokenizer.vocab_size).all():
            answer_input_ids[answer_input_ids >= self.tokenizer.vocab_size] = self.tokenizer.unk_token_id
        if not (answer_input_ids >= 0).all():
            answer_input_ids[answer_input_ids < 0] = self.tokenizer.unk_token_id

        return {
            'query': {
                'input_ids': query_input_ids,
                'attention_mask': query_enc['attention_mask'][0],
            },
            'answer': {
                'input_ids': answer_input_ids,
                'attention_mask': answer_enc['attention_mask'][0],
            }
        }

    def get_tokenized_item(self, idx: int, episode: dict = None) -> dict:
        if self.is_pre_tokenized:
            return self.inputs[idx]

        item = self.episodes[idx] if episode is None else episode
        interactions = item[self.interactions_field]

        # Tokenize all interactions
        tokenized_interactions = [
            self._tokenize_interaction(inter) for inter in interactions
        ]

        # First interaction provides initial query/answer
        if tokenized_interactions:
            first = tokenized_interactions[0]
            follow_ups = tokenized_interactions[1:] if len(tokenized_interactions) > 1 else []

            return {
                'query': first['query'],
                'answer': first['answer'],
                'interactions': follow_ups,
            }
        else:
            # Empty episode - return dummy data
            empty_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            empty_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            return {
                'query': {'input_ids': empty_ids, 'attention_mask': empty_mask},
                'answer': {'input_ids': empty_ids.clone(), 'attention_mask': empty_mask.clone()},
                'interactions': [],
            }

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

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "HybridReasoningSmatDataset":
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
            **kwargs
        )

    @classmethod
    def concat_from_hf_hub(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            split: str = 'train',
            interactions_field: str = 'interactions',
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            **kwargs
    ):
        """Load and concatenate multiple datasets from HuggingFace Hub."""
        if load_kwargs is None:
            load_kwargs = {}

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]

        hf_dataset = concatenate_datasets(hf_datasets)

        return cls(
            hf_dataset,
            tokenizer,
            max_seq_len=max_seq_len,
            interactions_field=interactions_field,
            **kwargs
        )

    @classmethod
    def concat_from_hf_hub_with_subset(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            split: str = 'train',
            interactions_field: str = 'interactions',
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            valid_size: Union[float, tuple[int]] = 0.1,
            **kwargs
    ):
        """Load and concatenate multiple datasets with validation split."""
        if load_kwargs is None:
            load_kwargs = {}

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in
            zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]

        if isinstance(valid_size, float):
            hf_ds_dicts = [dataset.train_test_split(test_size=valid_size) for dataset in hf_datasets]
        else:
            hf_ds_dicts = [dataset.train_test_split(test_size=valid_size[i]) for i, dataset in enumerate(hf_datasets)]

        hf_dataset = concatenate_datasets([ds_dict['train'] for ds_dict in hf_ds_dicts])
        hf_valid_dataset = concatenate_datasets([ds_dict['test'] for ds_dict in hf_ds_dicts])

        return (
            cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, interactions_field=interactions_field, **kwargs),
            cls(hf_valid_dataset, tokenizer, max_seq_len=max_seq_len, interactions_field=interactions_field, **kwargs)
        )

    @staticmethod
    def collate_hybrid_smat_batch(batch: list[dict]) -> dict:
        """Collate function for HybridReasoningSmatDataset with nested interactions."""

        def collate_interaction_batch(interaction_batch: list) -> dict[str, dict[str, torch.Tensor]]:
            """Helper to collate a batch of interactions."""
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

        batch_interactions = [x['interactions'] for x in batch]

        # Handle variable-length interaction lists by padding
        max_interactions = max(len(inters) for inters in batch_interactions)

        if max_interactions > 0:
            # Pad shorter interaction lists with dummy interactions
            padded_interactions = []
            for inters in batch_interactions:
                if len(inters) < max_interactions:
                    # Create dummy interactions for padding
                    dummy = inters[0] if inters else {
                        'query': {
                            'input_ids': torch.zeros_like(batch[0]['query']['input_ids']),
                            'attention_mask': torch.zeros_like(batch[0]['query']['attention_mask']),
                        },
                        'answer': {
                            'input_ids': torch.zeros_like(batch[0]['answer']['input_ids']),
                            'attention_mask': torch.zeros_like(batch[0]['answer']['attention_mask']),
                        }
                    }
                    padded = list(inters) + [dummy] * (max_interactions - len(inters))
                    padded_interactions.append(padded)
                else:
                    padded_interactions.append(inters)

            transposed_interactions = list(zip(*padded_interactions))

            return {
                **collate_interaction_batch(batch),
                'interactions': [
                    collate_interaction_batch(list(step_batch)) for step_batch in transposed_interactions
                ]
            }
        else:
            return {
                **collate_interaction_batch(batch),
                'interactions': []
            }
