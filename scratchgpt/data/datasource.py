from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources that provide text samples for training."""

    def __iter__(self) -> Iterator[str]:
        """Iterate over text samples in the dataset."""
        ...

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> str:
        """Get a specific text sample by index."""
        ...

    def map(
        self,
        function: Callable[..., Any],
        batched: bool = False,
        num_proc: int | None = None,
        remove_columns: list[str] | None = None,
        **kwargs: Any,
    ) -> "DataSource":
        """Apply a function to all samples in the dataset."""
        ...

    def train_test_split(
        self,
        test_size: float | None = None,
        train_size: float | None = None,
        seed: int | None = None,
    ) -> dict[str, "DataSource"]:
        """Split dataset into train and test sets."""
        ...

    def select(self, indices: list[int]) -> "DataSource":
        """Select specific indices from the dataset."""
        ...

    @property
    def column_names(self) -> list[str] | None:
        """Return column names if dataset has multiple columns."""
        ...
