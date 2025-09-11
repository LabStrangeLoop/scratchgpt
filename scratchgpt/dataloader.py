from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PretokenizedDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        token_file: Path,
        block_size: int,
        dtype: np.dtype,
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
