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
from scratchgpt.metering import AverageValueMeter
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer.base_tokenizer import Tokenizer


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

    def _run_epoch(self, dataloader: DataLoader[dict[str, Tensor]], stage: str) -> float:
        """Runs a single epoch of training or validation."""
        is_train = stage == "train"
        self.model.train(is_train)
        meter = AverageValueMeter()

        pbar = tqdm(dataloader, desc=stage.capitalize(), file=sys.stdout)
        with torch.set_grad_enabled(is_train):
            for batch in pbar:
                # IMPORTANT: implicit strong relationship to tokenize_utils.py
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(input_ids)
                B, T, C = logits.shape
                loss: Tensor = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

                if is_train:
                    loss.backward()  # type: ignore[no-untyped-call]
                    self.optimizer.step()

                meter.add(loss.item())
                mean, std = meter.value()
                pbar.set_postfix_str(f"Loss: {mean:.4f} ± {std:.4f}", refresh=True)

        mean_loss, std_loss = meter.value()
        print(f"📈 **{stage.capitalize()} Loss:** {mean_loss:.4f} ± {std_loss:.4f}")

        return mean_loss

    def train(
        self,
        data_source: DataSource,
        tokenizer: Tokenizer,
    ) -> None:
        """
        Trains the model.

        This method orchestrates the entire training pipeline, using HF datasets
        for efficient tokenization caching and data loading.
        """
        train_loader, val_loader = data_source.get_dataloaders(
            tokenizer=tokenizer,
            block_size=self.model._block_size,
            batch_size=self.config.batch_size,
            splits=self.config.splits,
            random_seed=self.config.random_seed,
        )

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
            else:
                print("🎉 Saving latest model as best because we have no validation dataset")
                torch.save(self.model.state_dict(), best_model_path)
