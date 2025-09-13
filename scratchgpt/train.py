import argparse
import sys
from pathlib import Path

import torch
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch.optim import AdamW

try:
    from scratchgpt.tokenizer.hf_tokenizer import HuggingFaceTokenizer
except ImportError:
    print(
        "HuggingFaceTokenizer not available. Please install the hf-tokenizers extras:\n"
        "pip install 'scratchgpt[hf-tokenizers]'",
        file=sys.stderr,
    )
    sys.exit(1)

from scratchgpt.config import ScratchGPTConfig
from scratchgpt.data.datasource import DataSource, FileDataSource, FolderDataSource
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.model_io import load_model, save_tokenizer
from scratchgpt.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Creates the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train a scratch-gpt model.")
    parser.add_argument(
        "-e",
        "--experiment",
        type=Path,
        required=True,
        help="The path to the experiment folder for saving checkpoints and configs.",
    )
    parser.add_argument(
        "-d",
        "--data_source",
        type=Path,
        required=True,
        help="The path to the training data source (file or folder).",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="gpt2",
        help="The name of the Hugging Face Hub tokenizer to use (e.g., 'gpt2', 'bert-base-uncased').",
    )
    parser.add_argument(
        "-dv",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="The hardware device to run training on.",
    )
    return parser.parse_args()


def get_data_source(path: Path) -> DataSource:
    """Instantiates the correct DataSource based on the path."""
    if path.is_file():
        print(f"Using single file data source: {path}")
        return FileDataSource(path)
    if path.is_dir():
        print(f"Using folder data source: {path}")
        return FolderDataSource(path)
    raise FileNotFoundError(f"Data source path not found or is not a file/directory: {path}")


def main() -> None:
    """Main script to configure and run the training process."""
    args = parse_args()
    args.experiment.mkdir(exist_ok=True, parents=True)

    # 1. Load or create the configuration
    config_path = args.experiment / "scratch_gpt.yaml"
    if config_path.exists():
        print(f"Loading existing config from {config_path}")
        config = parse_yaml_file_as(ScratchGPTConfig, config_path)
    else:
        print("No existing config found, creating a default one.")
        config = ScratchGPTConfig()

    torch.manual_seed(config.training.random_seed)

    # 2. Get the tokenizer from the Hugging Face Hub
    tokenizer = HuggingFaceTokenizer.from_hub(repo_id=args.tokenizer)
    config.architecture.vocab_size = tokenizer.vocab_size

    # 3. Instantiate the data sources
    data_source = get_data_source(args.data_source)

    # 4. Set up the model and optimizer
    device = torch.device(args.device)
    print(f"Using device: {device}")
    model = TransformerLanguageModel(config)

    # Load existing model weights if they exist in the experiment folder
    best_model_path = args.experiment / "best_model_weights.pth"
    model = load_model(best_model_path, model, device)

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)

    # 5. Instantiate the Trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        optimizer=optimizer,
        experiment_path=args.experiment,
        device=device,
    )

    # 6. Save the final config and tokenizer, then start training
    print("Saving configuration and tokenizer...")
    to_yaml_file(config_path, config)
    save_tokenizer(args.experiment, tokenizer)

    print("\nStarting training...")
    trainer.train(data=data_source, tokenizer=tokenizer)
    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
