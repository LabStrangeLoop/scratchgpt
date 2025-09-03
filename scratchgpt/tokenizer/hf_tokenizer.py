import json
from pathlib import Path
from typing import Self, cast, override

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as HFTokenizer

from .base_tokenizer import SerializableTokenizer, register_tokenizer


@register_tokenizer("HuggingFaceTokenizer")
class HuggingFaceTokenizer(SerializableTokenizer):
    """
    A wrapper around the Hugging Face `tokenizers` library that conforms to
    the SerializableTokenizer interface.
    """

    def __init__(self, tokenizer: HFTokenizer):
        self._tokenizer = tokenizer

    @override
    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text).ids)

    @override
    def decode(self, encoding: list[int]) -> str:
        return cast(str, self._tokenizer.decode(encoding))

    @property
    @override
    def vocab_size(self) -> int:
        return cast(int, self._tokenizer.get_vocab_size())

    @property
    @override
    def vocabulary(self) -> list[str]:
        # This can be slow for large vocabularies.
        vocab = self._tokenizer.get_vocab()
        # Sorting by token ID to ensure order
        return [item[0] for item in sorted(vocab.items(), key=lambda item: item[1])]

    @override
    def save(self, tokenizer_path: Path) -> None:
        """Saves the HF tokenizer and a config file."""
        super().save(tokenizer_path)

        hf_tokenizer_file = tokenizer_path / "tokenizer.json"
        self._tokenizer.save(str(hf_tokenizer_file))

        config = {
            "tokenizer_type": "HuggingFaceTokenizer",
            "hf_tokenizer_file": "tokenizer.json",
        }
        config_path = tokenizer_path / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    @override
    def load(cls, tokenizer_path: Path) -> Self:
        """Loads a HuggingFaceTokenizer from a directory."""
        hf_tokenizer_file = tokenizer_path / "tokenizer.json"
        if not hf_tokenizer_file.is_file():
            raise FileNotFoundError(f"Hugging Face tokenizer file not found at {hf_tokenizer_file}")

        hf_tokenizer = HFTokenizer.from_file(str(hf_tokenizer_file))
        return cls(tokenizer=hf_tokenizer)

    @classmethod
    def from_hub(cls, repo_id: str, filename: str = "tokenizer.json") -> Self:
        """
        Downloads and loads a tokenizer from the Hugging Face Hub.
        """
        # Download the tokenizer file from the Hub
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Instantiate the HF tokenizer and our wrapper
        hf_tokenizer = HFTokenizer.from_file(local_path)
        return cls(tokenizer=hf_tokenizer)
