import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scratchgpt.config import ScratchGPTTraining
from scratchgpt.data.datasource import ByteSizableDataSource, DataSource
from scratchgpt.dataloader import PretokenizedDataset
from scratchgpt.metering import AverageValueMeter
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer.base_tokenizer import Tokenizer


def get_dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    """Determine the smallest possible uint dtype for a given vocabulary size."""
    if vocab_size < 2**8:
        return np.dtype(np.uint8)
    if vocab_size < 2**16:
        return np.dtype(np.uint16)
    if vocab_size < 2**32:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


class Trainer:
    """Orchestrates the model training, validation, and checkpointing."""

    def __init__(
        self,
        model: TransformerLanguageModel,
        config: ScratchGPTTraining,
        optimizer: Optimizer,
        experiment_path: Path,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.experiment_path = experiment_path
        self.device = device
        self.experiment_path.mkdir(exist_ok=True, parents=True)

    def _pretokenize(
        self,
        data_source: DataSource,
        tokenizer: Tokenizer,
        output_path: Path,
        dtype: np.dtype,
    ) -> None:
        """Iterates through a DataSource, tokenizes it, and saves to a binary file."""
        total_size = None
        unit = "samples"
        # Check if we can provide a more detailed byte-level progress bar
        if isinstance(data_source, ByteSizableDataSource):
            total_size = data_source.total_bytes()
            unit = "B"

        with (
            open(output_path, "wb") as f,
            tqdm(total=total_size, unit=unit, unit_scale=True, desc="Tokenizing") as pbar,
        ):
            for text_sample in data_source:
                tokens = tokenizer.encode(text_sample)
                f.write(np.array(tokens, dtype=dtype).tobytes())

                if total_size:
                    pbar.update(len(text_sample.encode("utf-8", errors="ignore")))
                else:
                    pbar.update(1)

    def _get_dataloader(self, data_source: DataSource, tokenizer: Tokenizer, cache_file: Path) -> DataLoader:
        """Handles DataLoader creation, using a pre-tokenized cache if it exists."""
        dtype = get_dtype_for_vocab_size(tokenizer.vocab_size)

        if not cache_file.exists():
            print(f"⏳ Cache file not found. Pre-tokenizing data to '{cache_file}'...")
            self._pretokenize(data_source, tokenizer, cache_file, dtype)

        print(f"✅ Loading pre-tokenized data from '{cache_file}'")
        dataset = PretokenizedDataset(
            token_file=cache_file,
            block_size=self.model._block_size,
            dtype=dtype,
        )
        # num_workers can be configured or determined dynamically
        cpu_count = torch.multiprocessing.cpu_count()
        num_workers = int(cpu_count / 2) if cpu_count else 4
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

    def _run_epoch(self, dataloader: DataLoader, stage: str) -> float:
        """Runs a single epoch of training or validation."""
        is_train = stage == "train"
        self.model.train(is_train)
        meter = AverageValueMeter()

        pbar = tqdm(dataloader, desc=stage.capitalize(), file=sys.stdout)
        with torch.set_grad_enabled(is_train):
            for batch, targets in pbar:
                batch, targets = batch.to(self.device), targets.to(self.device)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(batch)
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                meter.add(loss.item())
                mean, std = meter.value()
                pbar.set_postfix_str(f"Loss: {mean:.4f} ± {std:.4f}", refresh=True)

        mean_loss, std_loss = meter.value()
        print(f"📈 **{stage.capitalize()} Loss:** {mean_loss:.4f} ± {std_loss:.4f}")

        return mean_loss

    def train(
        self,
        train_data: DataSource,
        tokenizer: Tokenizer,
        val_data: DataSource | None = None,
    ):
        """
        Trains the model.

        This method orchestrates the entire training pipeline, including optional
        data pre-tokenization, executing training and validation epochs, and
        saving model checkpoints.
        """
        train_cache = self.experiment_path / "train_data.bin"
        train_loader = self._get_dataloader(train_data, tokenizer, train_cache)

        val_loader = None
        if val_data:
            val_cache = self.experiment_path / "val_data.bin"
            val_loader = self._get_dataloader(val_data, tokenizer, val_cache)

        best_val_loss = float("inf")
        latest_model_path = self.experiment_path / "latest_model_weights.pth"
        best_model_path = self.experiment_path / "best_model_weights.pth"

        for epoch in range(self.config.max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.max_epochs} ---")
            self._run_epoch(train_loader, "train")
            torch.save(self.model.state_dict(), latest_model_path)

            if val_loader:
                val_loss = self._run_epoch(val_loader, "validation")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"🎉 New best validation loss: {best_val_loss:.4f}. Saving model...")
                    torch.save(self.model.state_dict(), best_model_path)
