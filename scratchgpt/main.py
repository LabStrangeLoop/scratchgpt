import argparse
import os
import sys
from typing import Literal

import torch
from pydantic_yaml import to_yaml_file
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.types import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ScratchGPTConfig
from .dataloader import FileTextProvider, FolderTextProvider, TextDataset, TextProvider
from .metering import AverageValueMeter
from .model.model import TransformerLanguageModel, print_model_complexity
from .model_io import (
    get_best_model_weights_path,
    get_latest_model_weights_path,
    get_tokenizer,
    load_model,
    save_tokenizer,
)

config = ScratchGPTConfig()

torch.manual_seed(config.training.random_seed)


def parse_args() -> argparse.Namespace:
    """
    Create CLI args parser and execute it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_source",
        help="The file you want to train on",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        help="The path to the folder where to save experiment checkpoints",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="What hardware you want to run the model on",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    return parser.parse_args()


DatasetType = tuple[Tensor, Tensor]


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[DatasetType],
    device: torch.device,
    stage: Literal["train", "validation", "test"],
    optimizer: Optimizer | None = None,
) -> tuple[float, float]:
    """
    Run a single epoch of training, validation, or testing.

    Args:
        model: The model to run the epoch on.
        dataloader: The DataLoader to use for the epoch.
        device: The device to run on (e.g., 'cuda' or 'cpu').
        stage: The stage of the epoch ('train', 'validation', or 'test').
        optimizer: The optimizer to use for training (only used if stage is 'train').

    Returns:
        A tuple containing the mean and standard deviation of the loss for the epoch.
    """
    average_loss = AverageValueMeter()

    is_train = stage == "train"
    model.train(is_train)

    pbar = tqdm(total=len(dataloader), desc=stage.capitalize(), file=sys.stdout)

    with torch.set_grad_enabled(is_train):
        for batch, targets in dataloader:
            batch = batch.to(device)
            targets = targets.to(device)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits, loss = model(batch, targets)

            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()

            average_loss.add(loss.item())

            mean, std = average_loss.value()
            pbar.set_description(f"📉 {stage.capitalize()} Loss mean: {mean:.4f}    std: {std:.4f}")
            pbar.update(1)

    pbar.close()
    return average_loss.value()


def get_text_provider(path: str) -> TextProvider:
    if os.path.isdir(path):
        return FolderTextProvider(path)
    return FileTextProvider(path)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using the device: {device}")

    text_provider = get_text_provider(args.train_source)

    tokenizer = get_tokenizer(args.experiment)
    config.architecture.vocab_size = tokenizer.vocab_size
    print(config)

    train_dataset = TextDataset(text_provider, tokenizer, config.architecture.block_size, "train", 0.9)
    val_dataset = TextDataset(text_provider, tokenizer, config.architecture.block_size, "validation", 0.1)

    cpu_count = os.cpu_count() or 4
    train_dataloader = DataLoader(
        train_dataset,
        config.training.batch_size,
        pin_memory=True,
        num_workers=int(cpu_count / 2),
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        config.training.batch_size,
        pin_memory=True,
        num_workers=int(cpu_count / 2),
        shuffle=False,
    )

    best_model_path = get_best_model_weights_path(args.experiment)
    latest_model_path = get_latest_model_weights_path(args.experiment)

    model = TransformerLanguageModel(
        config=config,
        device=device,
    )
    model = load_model(best_model_path, model, device)

    print_model_complexity(model, config, device)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)

    best_val_loss = float("inf")

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment, exist_ok=True)

    save_tokenizer(args.experiment, tokenizer)
    model_config = f"{args.experiment}/scratch_gpt.yaml"
    print(f"Saving this models config to {model_config}")
    to_yaml_file(model_config, config)

    try:
        for epoch in range(config.training.max_epochs):
            print(f"Epoch {epoch + 1}/{config.training.max_epochs}")

            train_loss_mean, train_loss_std = run_epoch(
                model=model,
                dataloader=train_dataloader,
                device=device,
                stage="train",
                optimizer=optimizer,
            )
            print(f"Training Loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}")
            torch.save(model.state_dict(), latest_model_path)

            val_loss_mean, val_loss_std = run_epoch(
                model=model,
                dataloader=val_dataloader,
                device=device,
                stage="validation",
            )
            print(f"Validation Loss: {val_loss_mean:.4f} ± {val_loss_std:.4f}")

            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                print(f"Saving new best model @ {best_model_path} with validation loss: {val_loss_mean:.4f}")
                torch.save(model.state_dict(), best_model_path)

            print()
    except KeyboardInterrupt:
        torch.save(model.state_dict(), latest_model_path)
        print("Trying my best here")

    prompt = input("Tell me your prompt: ")
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=500)
    first_batch_trained = tokenizer.decode(generated[0].tolist())
    print(first_batch_trained)


if __name__ == "__main__":
    main()
