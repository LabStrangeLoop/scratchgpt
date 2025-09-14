from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, IterableDataset, load_dataset

from scratchgpt.data.datasource import DataSource


class HFDataSource:
    """DataSource implementation using HuggingFace datasets library."""

    def __init__(
        self,
        path_or_name: str | Path,
        split: str = "train",
        streaming: bool = False,
        text_column: str = "text",
        **load_kwargs: Any,
    ):
        """
        Initialize HF DataSource.

        Args:
            path_or_name: Dataset name from Hub, or path to local file/folder
            split: Which split to use (train, validation, test)
            streaming: Whether to use streaming mode for large datasets
            text_column: Name of the column containing text data
            **load_kwargs: Additional arguments for load_dataset
        """
        self.path_or_name = str(path_or_name)
        self.split = split
        self.streaming = streaming
        self.text_column = text_column
        self._dataset = self._load_dataset(**load_kwargs)

    def _load_dataset(self, **load_kwargs: Any) -> Dataset | IterableDataset:
        """Load dataset from Hub or local files."""
        path = Path(self.path_or_name)

        # Check if it's a local path
        if path.exists():
            if path.is_file():
                # Single file
                extension = path.suffix.lower()
                if extension == ".csv":
                    dataset = load_dataset(
                        "csv", data_files=str(path), split=self.split, streaming=self.streaming, **load_kwargs
                    )
                elif extension in [".json", ".jsonl"]:
                    dataset = load_dataset(
                        "json", data_files=str(path), split=self.split, streaming=self.streaming, **load_kwargs
                    )
                elif extension == ".parquet":
                    dataset = load_dataset(
                        "parquet", data_files=str(path), split=self.split, streaming=self.streaming, **load_kwargs
                    )
                else:
                    # Default to text format
                    dataset = load_dataset(
                        "text", data_files=str(path), split=self.split, streaming=self.streaming, **load_kwargs
                    )
            else:
                # Directory with text files
                data_files = list(path.glob("**/*.txt")) + list(path.glob("**/*.md"))
                if not data_files:
                    # Try other text extensions
                    data_files = list(path.glob("**/*"))
                    data_files = [f for f in data_files if f.is_file()]

                data_files_str = [str(f) for f in data_files]
                dataset = load_dataset(
                    "text", data_files=data_files_str, split=self.split, streaming=self.streaming, **load_kwargs
                )
        else:
            # Assume it's a Hub dataset
            dataset = load_dataset(self.path_or_name, split=self.split, streaming=self.streaming, **load_kwargs)

        return cast(Dataset | IterableDataset, dataset)

    def __iter__(self) -> Iterator[str]:
        """Iterate over text samples."""
        for sample in self._dataset:
            if isinstance(sample, dict):
                yield sample[self.text_column]
            else:
                yield str(sample)

    def __len__(self) -> int:
        """Return dataset length."""
        if isinstance(self._dataset, IterableDataset):
            # Streaming datasets don't have length
            raise TypeError("Streaming datasets don't support len()")
        return len(self._dataset)

    def __getitem__(self, idx: int) -> str:
        """Get sample by index."""
        if isinstance(self._dataset, IterableDataset):
            raise TypeError("Streaming datasets don't support indexing")
        sample = self._dataset[idx]
        if isinstance(sample, dict):
            return sample[self.text_column]
        return str(sample)

    def map(
        self,
        function: Callable[..., Any],
        batched: bool = False,
        num_proc: int | None = None,
        remove_columns: list[str] | None = None,
        **kwargs: Any,
    ) -> DataSource:
        """Apply function to dataset samples."""
        mapped_dataset = self._dataset.map(
            function,
            batched=batched,
            num_proc=num_proc,
            remove_columns=remove_columns,
            **kwargs,
        )
        new_source = HFDataSource(self.path_or_name, self.split, self.streaming, self.text_column)
        new_source._dataset = mapped_dataset
        return new_source

    def train_test_split(
        self,
        test_size: float | None = None,
        train_size: float | None = None,
        seed: int | None = None,
    ) -> dict[str, DataSource]:
        """Split into train and test sets."""
        if isinstance(self._dataset, IterableDataset):
            raise TypeError("Streaming datasets don't support train_test_split")

        split_dataset = self._dataset.train_test_split(
            test_size=test_size,
            train_size=train_size,
            seed=seed,
        )

        result: dict[str, DataSource] = {}
        for split_name, split_data in split_dataset.items():
            new_source = HFDataSource(self.path_or_name, self.split, self.streaming, self.text_column)
            new_source._dataset = split_data
            result[split_name] = new_source

        return result

    def select(self, indices: list[int]) -> DataSource:
        """Select specific indices."""
        if isinstance(self._dataset, IterableDataset):
            raise TypeError("Streaming datasets don't support select")

        selected = self._dataset.select(indices)
        new_source = HFDataSource(self.path_or_name, self.split, self.streaming, self.text_column)
        new_source._dataset = selected
        return new_source

    @property
    def column_names(self) -> list[str] | None:
        """Get column names if available."""
        if hasattr(self._dataset, "column_names"):
            return self._dataset.column_names
        return None

    @property
    def dataset(self) -> Dataset | IterableDataset:
        """Access underlying HF dataset."""
        return self._dataset
