import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, BertPreTokenizer, ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from transformers import PreTrainedTokenizerFast
from typing import Any

class TokenizerTrainer:
    def __init__(
            self,
            vocab_size: int = 30000,
            model_type: str = "byte-level-bpe",  # Options: "bpe", "wordpiece", "unigram", "sentencepiece"
            special_tokens: list[str] = None,
            lowercase: bool = False,
            normalization: bool = False,
            pre_tokenizer_type: str = "bert",  # Options: "bert", "whitespace_punctuation",
            vocab: Any = None,
            byte_fallback: bool = False,
            max_input_chars_per_word: int = 32,
            use_post_processor: bool = True,
            post_processor_single: str = "[BOS] $A [EOS]",
            post_processor_pair: str = "[BOS] $A [EOS][BOS] $B:1 [EOS]:1",
            post_processor_special_tokens: list[str] = None,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else ["[PAD]", "[UNK]", "[BOS]", "[EOS]",
                                                                                 "[MASK]"]
        self.model_type = model_type.lower()
        self.lowercase = lowercase
        self.normalization = normalization
        self.pre_tokenizer_type = pre_tokenizer_type.lower()

        # Initialize tokenizer model
        if self.model_type == "bpe":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]", vocab=vocab, byte_fallback=byte_fallback))
        elif self.model_type == "wordpiece":
            self.tokenizer = Tokenizer(
                WordPiece(unk_token="[UNK]", vocab=vocab, max_input_chars_per_word=max_input_chars_per_word))
        elif self.model_type == "unigram":
            self.tokenizer = Tokenizer(Unigram(unk_id="[UNK]", vocab=None, byte_fallback=byte_fallback))
        elif self.model_type == "wordlevel":
            self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]", vocab=None))
        elif self.model_type == "byte-level-bpe":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]", vocab=vocab, byte_fallback=byte_fallback))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Configure pre-tokenizer
        if self.model_type != "byte-level-bpe":
            if self.pre_tokenizer_type == "bert":
                self.tokenizer.pre_tokenizer = BertPreTokenizer()
            elif self.pre_tokenizer_type == "whitespace_punctuation":
                self.tokenizer.pre_tokenizer = Whitespace()
                self.tokenizer.pre_tokenizer = Punctuation()
            elif self.pre_tokenizer_type == "whitespace":
                self.tokenizer.pre_tokenizer = Whitespace()
            else:
                raise ValueError(f"Unsupported pre-tokenizer: {pre_tokenizer_type}")

        # Add normalization steps
        if self.normalization:
            normalizers = []
            if self.lowercase:
                normalizers.append(Lowercase())
            normalizers.append(NFKC())
            self.tokenizer.normalizer = Sequence(normalizers)

        self.use_post_processor = use_post_processor
        self.post_processor_single = post_processor_single
        self.post_processor_pair = post_processor_pair
        self.post_processor_special_tokens = post_processor_special_tokens

    def train(
            self,
            files: list[str],
            limit_alphabet: int = 1000,
            show_progress: bool = True,
            **kwargs
    ):
        # Prepare trainer based on model type
        trainer_kwargs = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "limit_alphabet": limit_alphabet,
            "show_progress": show_progress,
            **kwargs  # Allow custom parameters
        }

        if self.model_type in ["bpe", "byte-level-bpe"]:
            trainer = BpeTrainer(**trainer_kwargs)
        elif self.model_type == "wordpiece":
            trainer = WordPieceTrainer(**trainer_kwargs)
        elif self.model_type == "unigram":
            trainer = UnigramTrainer(**trainer_kwargs)
        elif self.model_type == "wordlevel":
            trainer = WordLevelTrainer(**trainer_kwargs)

        # Train tokenizer
        self.tokenizer.train(files, trainer)

        if self.use_post_processor:
            post_processor_special_tokens = self.post_processor_special_tokens or ["[BOS]", "[EOS]"]
            self.tokenizer.post_processor = TemplateProcessing(
                single=self.post_processor_single,
                pair=self.post_processor_pair,
                special_tokens=[(token, self.tokenizer.token_to_id(token)) for token in post_processor_special_tokens],
            )

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save(f"{output_dir}/tokenizer.json")

    def load(self, model_path: str):
        self.tokenizer = Tokenizer.from_file(model_path)

    def get_hf_tokenizer(self):
        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )

    def push_to_hub(
            self,
            repo_id: str,
            create: bool = False,
            private: bool = False,
            api_token: str = None,
            **kwargs
    ):
        """
        Push the trained tokenizer to HuggingFace Hub.

        Args:
            repo_id (str): Hub repository name (e.g., "username/my-tokenizer")
            private (bool): Whether the repo is private
            api_token (str): HuggingFace API token (optional if already logged in)
            **kwargs: Additional args for HuggingFace API
        """
        from huggingface_hub import HfApi, Repository

        # Create a temporary directory for Hub upload
        temp_dir = "temp_hub_upload"
        os.makedirs(temp_dir, exist_ok=True)
        self.save(temp_dir)  # Save tokenizer files locally

        # Push to Hub using HuggingFace API
        api = HfApi(token=api_token)
        if create:
            api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
            )

        # Push files to the repo
        api.upload_folder(
            repo_id=repo_id,
            folder_path=temp_dir,
            repo_type="model",
            **kwargs
        )

        # Cleanup
        os.remove(Path(temp_dir) / 'tokenizer.json')
        os.rmdir(temp_dir)

    @staticmethod
    def hf_tokenizer_from_file(path: str):
        return PreTrainedTokenizerFast(
            tokenizer_file=path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        """
        Load tokenizer from HuggingFace Hub.

        Args:
            repo_id (str): Hub repository name (e.g., "username/my-tokenizer")
            **kwargs: Additional args for HuggingFace API
        """
        from huggingface_hub import hf_hub_download

        # Download tokenizer.json from Hub
        tokenizer_file = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            **kwargs
        )

        # Initialize trainer and load tokenizer
        trainer = cls()
        trainer.load(tokenizer_file)
        return trainer

def load_tokenizer_from_hf_hub(repo_id: str, **kwargs) -> PreTrainedTokenizerFast:
    return TokenizerTrainer.from_pretrained(repo_id, **kwargs).get_hf_tokenizer()

def load_tokenizer_from_file(path: str) -> PreTrainedTokenizerFast:
    return TokenizerTrainer.hf_tokenizer_from_file(path)
