import argparse
import sys
from pathlib import Path

import torch
from pydantic_yaml import parse_yaml_file_as
from rich.pretty import pprint as rpprint

from scratchgpt.config import ScratchGPTConfig

from .model.model import TransformerLanguageModel
from .model_io import get_best_model_weights_path, load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    """
    Create CLI arg parser and execute it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        help="What hardware you want to run the model on",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "-e",
        "--experiment",
        help="The path to the folder where to save experiment checkpoints",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--max_tokens",
        type=int,
        default=256,
        help="Number of tokens you want the model produce",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_file: Path = args.experiment / "scratch_gpt.yaml"
    config = parse_yaml_file_as(ScratchGPTConfig, config_file)
    print(f"Using config file {config_file}")
    rpprint(config.model_dump(), indent_guides=True, expand_all=True)

    tokenizer = load_tokenizer(args.experiment)

    device = torch.device(args.device)
    best_model_path = get_best_model_weights_path(args.experiment)

    model = TransformerLanguageModel(
        config=config,
    )
    load_model(best_model_path, model, device)

    while True:
        try:
            prompt = input("Tell me your prompt: ")
            if prompt == "quit":
                sys.exit(0)

            context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
            generated = model.generate(context, max_new_tokens=args.max_tokens)
            inferred = tokenizer.decode(generated[0].tolist())
            print(inferred)
            print("-----------------------------------")
        except (EOFError, SystemExit, KeyboardInterrupt):
            print("\n", "=" * 20, "Goodbye", "=" * 20)
            break


if __name__ == "__main__":
    main()
