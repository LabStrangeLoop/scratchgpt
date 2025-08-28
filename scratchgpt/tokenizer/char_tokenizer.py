from typing import Any, override

from .base_tokenizer import Tokenizer


def get_vocab(text: str) -> list[str]:
    chars = sorted(set(text))
    return chars


def str_to_int(chars: list[str]) -> dict[str, int]:
    return {char: idx for idx, char in enumerate(chars)}


def int_to_str(chars: list[str]) -> dict[int, str]:
    return dict(enumerate(chars))


class CharTokenizer(Tokenizer):

    def __init__(self, text: str) -> None:
        self._vocabulary = get_vocab(text)
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
        return "".join(self._decoding_mapping[v] for v in encoding)


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
