import json
from pathlib import Path
from typing import Any, Self, override

from .base_tokenizer import SerializableTokenizer, Tokenizer, register_tokenizer


def get_vocab(text: str) -> list[str]:
    chars = sorted(set(text))
    return chars


def str_to_int(chars: list[str]) -> dict[str, int]:
    return {char: idx for idx, char in enumerate(chars)}


def int_to_str(chars: list[str]) -> dict[int, str]:
    return dict(enumerate(chars))


@register_tokenizer("CharTokenizer")
class CharTokenizer(SerializableTokenizer):
    def __init__(self, text: str | None = None, vocab: list[str] | None = None) -> None:
        if text is not None:
            self._vocabulary = get_vocab(text)
        elif vocab is not None:
            self._vocabulary = vocab
        else:
            raise ValueError("Either 'text' or 'vocab' must be provided.")

        self._encoding_mapping = str_to_int(self._vocabulary)
        self._decoding_mapping = int_to_str(self._vocabulary)

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    @override
    def vocabulary(self) -> list[Any]:
        return self._vocabulary

    @override
    def encode(self, text: str) -> list[int]:
        return [self._encoding_mapping[char] for char in text]

    @override
    def decode(
        self,
        encoding: list[int],
    ) -> str:
        return "".join(self._decoding_mapping.get(v, "?") for v in encoding)

    @override
    def save(self, tokenizer_path: Path) -> None:
        """Saves the character vocabulary and config file."""
        super().save(tokenizer_path)

        # Save the vocabulary
        vocab_file = tokenizer_path / "vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, indent=2)

        # Save the metadata config file
        config = {
            "tokenizer_type": "CharTokenizer",
            "vocab_file": "vocab.json",
        }
        config_path = tokenizer_path / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    @override
    def load(cls, tokenizer_path: Path) -> Self:
        """Loads a CharTokenizer from a directory."""
        vocab_file = tokenizer_path / "vocab.json"
        if not vocab_file.is_file():
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")

        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)

        return cls(vocab=vocab)


class Utf8Tokenizer(Tokenizer):
    def __init__(self) -> None:
        self._vocabulary = list(range(256))

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    @override
    def vocabulary(self) -> list[Any]:
        return self._vocabulary

    @override
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @override
    def decode(
        self,
        encoding: list[int],
    ) -> str:
        bs = bytes(encoding)
        return "".join(bs.decode("utf-8"))
