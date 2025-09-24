from typing import Literal, Protocol

from scratchgpt.core.types import DictTensorLoader
from scratchgpt.tokenizer.base_tokenizer import Tokenizer


class DataSource(Protocol):
    """
    A protocol for classes that can provide training and validation DataLoaders.

    This uses structural subtyping. Any class that implements a matching
    `get_dataloaders` method will be considered a valid DataSource.
    """

    def get_dataloaders(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        batch_size: int,
        splits: tuple[float, float],
        random_seed: int,
        iteration_type: Literal["chunking", "sliding"],
    ) -> tuple[DictTensorLoader, DictTensorLoader | None]:
        """
        Processes data and returns train and validation DataLoaders.
        """
        ...
