import argparse
import math
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from rich.pretty import pprint as rpprint
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.types import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from scratchgpt.preprocess import File2FileTokenizerPreprocessor, FilePreprocessor, Folder2FileTokenizerPreprocessor
from scratchgpt.tokenizer.base_tokenizer import Tokenizer

from .config import ScratchGPTConfig
from .dataloader import PretokenizedDataset
from .metering import AverageValueMeter
from .model.model import TransformerLanguageModel, print_model_complexity
from .model_io import (
    get_best_model_weights_path,
    get_latest_model_weights_path,
    get_tokenizer,
    load_model,
    save_tokenizer,
)

DatasetType = tuple[Tensor, Tensor]


def parse_splits(value: str) -> list[float]:
    """
    Custom argparse type to validate and parse training splits.
    Splits should be provided as a semicolon-separated string of 3 floats
    (train, validation, test) that sum to 1.0.
    """
    try:
        splits = [float(x) for x in value.split(";")]
        if len(splits) != 3:
            raise ValueError("Exactly three split values for train, validation, and test are required.")
        if not math.isclose(sum(splits), 1.0):
            raise ValueError(f"Split values must sum to 1.0, but they sum to {sum(splits):.2f}.")
        return splits
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid split format '{value}'. Use 'train;val;test' format (e.g., '0.8;0.1;0.1'). Error: {e}"
        ) from e


def parse_args() -> argparse.Namespace:
    """
    Create CLI args parser and execute it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_source",
        help="The file or folder you want to train on",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        help="The path to the folder where to save experiment checkpoints",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="NumPy dtype for pre-tokenized .bin files (e.g., 'uint16'). Required if using a .bin file.",
    )
    return parser.parse_args()


def load_or_create_config(experiment_path: Path) -> ScratchGPTConfig:
    """
    Load config from experiment folder if it exists, otherwise create default.
    """
    config_path: Path = experiment_path / "scratch_gpt.yaml"

    if config_path.exists():
        print(f"Loading existing config from {config_path}")
        return parse_yaml_file_as(ScratchGPTConfig, config_path)
    else:
        print("No existing config found, creating default config")
        return ScratchGPTConfig()


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

            logits = model(batch)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss: Tensor = F.cross_entropy(logits, targets)

            if is_train and optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

            average_loss.add(loss.item())

            mean, std = average_loss.value()
            pbar.set_description(f"📉 {stage.capitalize()} Loss mean: {mean:.4f}    std: {std:.4f}")
            pbar.update(1)

    pbar.close()
    return average_loss.value()


def get_dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    """Determine the smallest possible uint dtype for a given vocabulary size."""
    if vocab_size < 2**8:
        return np.dtype(np.uint8)
    if vocab_size < 2**16:
        return np.dtype(np.uint16)
    if vocab_size < 2**32:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def prepare_dataset(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    config: ScratchGPTConfig,
) -> Dataset[tuple[Tensor, Tensor]]:
    """
    Prepare the dataset for training.
    - If the source is a .bin file, it loads it directly.
    - If the source is text, it preprocesses and caches it in the experiment folder.
    - If a cached version exists, it uses that instead of reprocessing.
    """
    cached_data_path = args.experiment / "preprocessed_data.bin"

    if args.train_source.suffix == ".bin":
        print(f"Loading pre-tokenized data directly from {args.train_source}")
        if not args.dtype:
            raise ValueError("--dtype must be specified when using a .bin file.")
        return PretokenizedDataset(
            token_file=args.train_source,
            block_size=config.architecture.block_size,
            dtype=np.dtype(args.dtype),
        )

    # For raw text, determine the best dtype based on the tokenizer's vocab size.
    dtype = get_dtype_for_vocab_size(tokenizer.vocab_size)

    if cached_data_path.exists():
        print(f"Found cached preprocessed data at {cached_data_path}. Loading it.")
        return PretokenizedDataset(
            token_file=cached_data_path,
            block_size=config.architecture.block_size,
            dtype=dtype,
        )

    print(f"No cached data found. Preprocessing '{args.train_source}' now.")
    if args.train_source.is_dir():
        preprocessor: FilePreprocessor = Folder2FileTokenizerPreprocessor(tokenizer)
    else:
        preprocessor = File2FileTokenizerPreprocessor(tokenizer)

    preprocessor(input_path=args.train_source, output_path=cached_data_path)

    print(f"Loading the newly preprocessed data from {cached_data_path}")
    return PretokenizedDataset(
        token_file=cached_data_path,
        block_size=config.architecture.block_size,
        dtype=dtype,
    )


def main() -> None:
    args = parse_args()

    config = load_or_create_config(args.experiment)

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment, exist_ok=True)

    torch.manual_seed(config.training.random_seed)
    print(f"Set random seed to: {config.training.random_seed}")

    device = torch.device(config.training.device)
    print(f"Using the device: {device}")

    tokenizer = get_tokenizer(args.experiment)
    config.architecture.vocab_size = tokenizer.vocab_size
    rpprint(config.model_dump(), indent_guides=True, expand_all=True)

    full_dataset = prepare_dataset(args, tokenizer, config)
    print(f"Splitting dataset into train/validation/test with ratios: {config.training.splits}")
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=full_dataset,
        lengths=config.training.splits,
        generator=torch.Generator().manual_seed(config.training.random_seed),
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print("Loading train, validation, and test loaders...")
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

    test_dataloader = None
    if len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            config.training.batch_size,
            pin_memory=True,
            num_workers=int(cpu_count / 2),
            shuffle=False,
        )

    print("Loaders initialized")

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

    if test_dataloader:
        print("\n--- Running Final Test Evaluation ---")
        print(f"Loading best model weights from {best_model_path}")
        model = load_model(best_model_path, model, device)

        test_loss_mean, test_loss_std = run_epoch(
            model=model,
            dataloader=test_dataloader,
            device=device,
            stage="test",
        )
        print("=" * 40)
        print(f"🔬 Final Test Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}")
        print("=" * 40)

    prompt = input("Tell me your prompt: ")
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=500)
    first_batch_trained = tokenizer.decode(generated[0].tolist())
    print(first_batch_trained)


if __name__ == "__main__":
    main()
