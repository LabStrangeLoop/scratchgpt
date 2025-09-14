import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scratchgpt.config import ScratchGPTTraining
from scratchgpt.data.datasource import DataSource
from scratchgpt.data.hf_datasource import HFDataSource
from scratchgpt.metering import AverageValueMeter
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer.base_tokenizer import Tokenizer
from scratchgpt.training.tokenize_utils import prepare_dataset_for_training


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

    def _get_dataloader(self, data_source: DataSource, tokenizer: Tokenizer) -> tuple[DataLoader, DataLoader]:
        """Handles DataLoader creation using HF datasets."""

        # Check if it's HFDataSource to access dataset directly
        if not isinstance(data_source, HFDataSource):
            raise TypeError("DataSource must be HFDataSource")

        print("⏳ Tokenizing dataset (using HF cached tokenization)...")

        # Get the underlying HF dataset
        dataset = data_source.dataset

        # Prepare dataset (tokenize and chunk)
        cpu_count = torch.multiprocessing.cpu_count()
        num_proc = max(1, int(cpu_count / 2))

        prepared_dataset = prepare_dataset_for_training(
            dataset,
            tokenizer,
            self.model._block_size,
            num_proc=num_proc,
        )

        print(f"✅ Dataset prepared with {len(prepared_dataset)} samples")

        # Split into train and validation
        train_split, val_split = self.config.splits

        # Use HF's train_test_split
        split_datasets = prepared_dataset.train_test_split(
            test_size=val_split,
            seed=self.config.random_seed,
        )

        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["test"]

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_proc,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_proc,
            drop_last=False,
        )

        return train_loader, val_loader

    def _run_epoch(self, dataloader: DataLoader, stage: str) -> float:
        """Runs a single epoch of training or validation."""
        is_train = stage == "train"
        self.model.train(is_train)
        meter = AverageValueMeter()

        pbar = tqdm(dataloader, desc=stage.capitalize(), file=sys.stdout)
        with torch.set_grad_enabled(is_train):
            for batch in pbar:
                # Extract input_ids and labels from batch
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(input_ids)
                B, T, C = logits.shape
                loss: Tensor = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

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
        data: DataSource,
        tokenizer: Tokenizer,
    ) -> None:
        """
        Trains the model.

        This method orchestrates the entire training pipeline, using HF datasets
        for efficient tokenization caching and data loading.
        """
        train_loader, val_loader = self._get_dataloader(data, tokenizer)

        best_val_loss = float("inf")
        latest_model_path = self.experiment_path / "latest_model_weights.pth"
        best_model_path = self.experiment_path / "best_model_weights.pth"

        for epoch in range(self.config.max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.max_epochs} ---")
            self._run_epoch(train_loader, "train")
            torch.save(self.model.state_dict(), latest_model_path)

            val_loss = self._run_epoch(val_loader, "validation")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🎉 New best validation loss: {best_val_loss:.4f}. Saving model...")
                torch.save(self.model.state_dict(), best_model_path)
