from functools import lru_cache
from typing import override

import tiktoken

from .base_tokenizer import Tokenizer


class TiktokenWrapper(Tokenizer):
    def __init__(self, encoding_name: str = "p50k_base"):
        """
        Initialize the TiktokenWrapper.

        Args:
            encoding_name (str): The name of the tiktoken encoding to use.
                Default is "cl100k_base" (used by gpt-3.5-turbo and gpt-4).
        """
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    @override
    def encode(self, text: str) -> list[int]:
        """Convert a string into a sequence of token IDs."""
        return self._tokenizer.encode(text)

    @override
    def decode(self, encoding: list[int]) -> str:
        """Convert a sequence of token IDs back into a string."""
        return self._tokenizer.decode(encoding)

    @property
    @override
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return self._tokenizer.max_token_value + 1

    @property
    @override
    def vocabulary(self) -> list[str]:
        """Return the learned vocabulary"""
        return list(self._get_cached_vocabulary())

    @lru_cache(maxsize=1)  # noqa: B019
    def _get_cached_vocabulary(self) -> tuple[str, ...]:
        """
        Cache and return the vocabulary as a tuple.

        This method is cached using lru_cache to avoid regenerating
        the vocabulary on every call.
        """
        return tuple(self._tokenizer.decode([i]) for i in range(self.vocab_size))

    def clear_cache(self) -> None:
        """Clear the cached vocabulary."""
        self._get_cached_vocabulary.cache_clear()


def main() -> None:
    tokenizer = TiktokenWrapper()

    # Test encoding and decoding
    text = "Hello, world! This is a test of the tiktoken wrapper."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Print first 10 vocabulary items as an example
    print("First 10 vocabulary items:")
    for token in tokenizer.vocabulary[:10]:
        print(token)


if __name__ == "__main__":
    main()
