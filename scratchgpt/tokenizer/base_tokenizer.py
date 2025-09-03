from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Self, TypeVar

T_SerializableTokenizer = TypeVar("T_SerializableTokenizer", bound="SerializableTokenizer")


TOKENIZER_REGISTRY: dict[str, type["SerializableTokenizer"]] = {}
"""
A simple registry to map tokenizer types to their classes.
This helps in dynamically loading the correct tokenizer.
"""


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string into a sequence of token IDs."""

    @abstractmethod
    def decode(self, encoding: list[int]) -> str:
        """Convert a sequence of token IDs back into a string."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""

    @property
    @abstractmethod
    def vocabulary(self) -> list[str]:
        """Return the learned vocabulary"""


class SerializableTokenizer(Tokenizer):
    """
    An extension of the Tokenizer ABC that adds methods for saving and loading.
    """

    @abstractmethod
    def save(self, tokenizer_path: Path) -> None:
        """
        Saves the tokenizer's state to a specified directory.

        This method should create a `tokenizer_config.json` with metadata
        and any other necessary data files (e.g., vocabulary).

        Args:
            tokenizer_path: The directory path to save the tokenizer to.
        """
        # Ensure the directory exists
        tokenizer_path.mkdir(exist_ok=True, parents=True)

    @classmethod
    @abstractmethod
    def load(cls, tokenizer_path: Path) -> Self:
        """
        Loads a tokenizer from a specified directory.

        Args:
            tokenizer_path: The directory containing the tokenizer's state.

        Returns:
            An instance of the tokenizer.
        """
        config_path = tokenizer_path / "tokenizer_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Tokenizer config not found at: {config_path}")

        raise NotImplementedError


def register_tokenizer(
    name: str,
) -> Callable[[type[T_SerializableTokenizer]], type[T_SerializableTokenizer]]:
    """
    A decorator to register a tokenizer class in the registry, preserving its type.
    """

    def decorator(cls: type[T_SerializableTokenizer]) -> type[T_SerializableTokenizer]:
        # Runtime check is still good practice.
        if not issubclass(cls, SerializableTokenizer):
            raise TypeError("Registered tokenizer must be a subclass of SerializableTokenizer.")
        TOKENIZER_REGISTRY[name] = cls
        return cls

    return decorator
