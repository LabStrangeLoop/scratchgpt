from abc import ABC, abstractmethod
import os
from typing import Literal, override

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .tokenizer.base_tokenizer import Tokenizer


class TextProvider(ABC):
    @abstractmethod
    def get_text(self) -> str:
        """This method fetches the text from the underlying storage"""


class FileTextProvider(TextProvider):
    def __init__(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise ValueError(f"File path {file_path} does not exist")

        self._data = ""
        with open(file_path, "r") as f:
            self._data = f.read()

    @override
    def get_text(self) -> str:
        return self._data


class FolderTextProvider(TextProvider):
    def __init__(self, dir_path: str) -> None:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory path {dir_path} does not exist")

        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory path {dir_path} is not a directory")

        self._data = ""
        for root, _, files in os.walk(dir_path):
            print(f"Loading data from {root}")
            for file_name in files:
                if not file_name.startswith("."):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._data += f.read() + "\n"  # Concatenate with a
                        # newline between files, could be the place to add
                        # special tokens

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
