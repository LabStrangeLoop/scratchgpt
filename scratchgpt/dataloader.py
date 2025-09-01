from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .tokenizer.base_tokenizer import Tokenizer


class TextProvider(ABC):
    @abstractmethod
    def get_text(self) -> str:
        """This method fetches the text from the underlying storage"""


class FileTextProvider(TextProvider):
    def __init__(self, file_path: Path) -> None:
        if not file_path.exists():
            raise ValueError(f"File path {file_path} does not exist")

        self._data = ""
        print(f"Loading data from {file_path}")
        with open(file_path) as f:
            self._data = f.read()
        print("Data Loaded")

    @override
    def get_text(self) -> str:
        return self._data


class FolderTextProvider(TextProvider):
    def __init__(self, dir_path: Path) -> None:
        if not dir_path.exists():
            raise ValueError(f"Directory path {dir_path} does not exist")

        if not dir_path.is_dir():
            raise ValueError(f"Directory path {dir_path} is not a directory")

        self._data = ""
        file_paths = list(dir_path.rglob("*"))
        print(f"Loading data from {dir_path}")
        for file_path in tqdm(file_paths, desc="Reading data files", unit="file"):
            if file_path.is_file() and not file_path.name.startswith("."):
                with open(file_path, encoding="utf-8") as f:
                    self._data += f.read() + "\n"

        print("Data Loaded")

    @override
    def get_text(self) -> str:
        return self._data


class TextDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        text_provider: TextProvider,
        tokenizer: Tokenizer,
        block_size: int,
        split: Literal["train", "validation", "test"],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.data = torch.tensor(self.tokenizer.encode(text_provider.get_text()), dtype=torch.long)

        total_size = len(self.data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        if split == "train":
            self.data = self.data[:train_size]
        elif split == "validation":
            self.data = self.data[train_size : train_size + val_size]
        elif split == "test":
            self.data = self.data[train_size + val_size :]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'.")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        block = self.data[idx : idx + self.block_size]
        target = self.data[idx + 1 : idx + self.block_size + 1]
        return block, target


class PretokenizedDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        token_file: Path,
        block_size: int,
        # Default is now an instance of the dtype class
        dtype: np.dtype = np.dtype(np.uint16),
    ) -> None:
        super().__init__()
        self.block_size = block_size

        all_tokens = np.memmap(token_file, dtype=dtype, mode="c")
        self.data = torch.from_numpy(all_tokens)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        block = self.data[idx : idx + self.block_size]
        target = self.data[idx + 1 : idx + self.block_size + 1]

        return block.long(), target.long()
