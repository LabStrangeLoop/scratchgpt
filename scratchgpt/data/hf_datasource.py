from collections.abc import Iterator
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from datasets import IterableDataset as HFIterableDataset
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

from scratchgpt.core.types import DictTensorLoader
from scratchgpt.tokenizer.base_tokenizer import Tokenizer
from scratchgpt.training.tokenize_utils import prepare_dataset_for_training


class _StreamingBlockDataset(TorchIterableDataset[dict[str, Tensor]]):
    """
    A PyTorch IterableDataset that wraps a Hugging Face IterableDataset,
    tokenizes it, and yields blocks of a fixed size.
    """

    def __init__(
        self,
        hf_dataset: HFIterableDataset,
        tokenizer: Tokenizer,
        block_size: int,
        text_column: str,
    ):
        self._hf_dataset = hf_dataset
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._text_column = text_column

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        buffer: list[int] = []
        chunk_size = self._block_size + 1

        for example in self._hf_dataset:
            # Safely get text from the example dict.
            if text := example.get(self._text_column):
                buffer.extend(self._tokenizer.encode(text))

                # Yield blocks as they become available in the buffer.
                while len(buffer) >= chunk_size:
                    block = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
                    yield {
                        "input_ids": torch.tensor(block[:-1]),
                        "labels": torch.tensor(block[1:]),
                    }


class HFDataSource:
    """
    A DataSource that loads data from the Hugging Face Hub.
    Handles both standard and streaming datasets.
    """

    def __init__(self, path_or_name: str, split: str = "train", streaming: bool = False, text_column: str = "text"):
        self._text_column = text_column
        self._streaming = streaming

        local_path = Path(path_or_name)

        # Case 1: The path is a local directory.
        if local_path.is_dir():
            print(f"Detected local directory. Loading all text files from: {local_path}")
            self._dataset = load_dataset(
                "text",
                data_dir=str(local_path),
                split=split,
                streaming=streaming,
            )
        # Case 2: The path is a local file.
        elif local_path.is_file():
            print(f"Detected local file: {local_path}")
            is_common_ext = local_path.suffix in {".txt", ".md"}

            if is_common_ext:
                print("Loading as plain text file.")
                self._dataset = load_dataset(
                    "text",
                    data_files={split: str(local_path)},
                    split=split,
                    streaming=streaming,
                )
            else:
                print("Attempting to infer file type...")
                self._dataset = load_dataset(
                    str(local_path),
                    split=split,
                    streaming=streaming,
                )
        # Case 3: The path is not local, assume it's a Hub dataset ID.
        else:
            print(f"Assuming '{path_or_name}' is a Hugging Face Hub dataset.")
            self._dataset = load_dataset(
                path_or_name,
                split=split,
                streaming=streaming,
            )

    def get_dataloaders(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        batch_size: int,
        splits: tuple[float, float],
        random_seed: int,
    ) -> tuple[DictTensorLoader, DictTensorLoader | None]:
        cpu_count = torch.multiprocessing.cpu_count() or 1
        num_proc = max(1, cpu_count // 2)

        match self._dataset:
            case Dataset():
                prepared_dataset = prepare_dataset_for_training(
                    self._dataset, tokenizer, block_size, self._text_column, num_proc
                )
                split_datasets = prepared_dataset.train_test_split(test_size=splits[1], seed=random_seed)
                train_loader = DataLoader(
                    split_datasets["train"],
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=num_proc,
                )
                val_loader = DataLoader(
                    split_datasets["test"],
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=num_proc,
                )
                return train_loader, val_loader

            case HFIterableDataset():
                print(
                    "⚠️  Note: Validation splitting is not supported for streaming datasets. "
                    "Validation loader will be None."
                )

                streaming_dataset = _StreamingBlockDataset(self._dataset, tokenizer, block_size, self._text_column)
                # shuffle=True is not supported for IterableDatasets in DataLoader
                train_loader = DataLoader(streaming_dataset, batch_size=batch_size)

                return train_loader, None

            case _:
                raise TypeError(f"Unsupported dataset type: {type(self._dataset)}")
