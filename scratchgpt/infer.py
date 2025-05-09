import argparse
import sys

import torch

from .main import TransformerLanguageModel
from .model_io import get_best_model_weights_path, get_tokenizer, load_model

BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_EPOCHS = 50
LEARNING_RATE = 3e-4
N_EMBED = 384
NUM_HEADS = 6
NUM_BLOCKS = 6


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
        type=str,
    )
    parser.add_argument(
        "-m",
        "--max_tokens",
        type=int,
        default=BLOCK_SIZE * 2,
        help="Number of tokens you want the model produce",
    )
    parser.add_argument("-t", "--tokenizer", help="please!", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(args.experiment, args.tokenizer)

    device = torch.device(args.device)
    best_model_path = get_best_model_weights_path(args.experiment)

    model = TransformerLanguageModel(NUM_HEADS, tokenizer.vocab_size, N_EMBED, BLOCK_SIZE, NUM_BLOCKS)
    load_model(best_model_path, model, device)

    while True:
        prompt = input("Tell me your prompt: ")
        if prompt == "quit":
            sys.exit(0)

        context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        generated = model.generate(context, max_new_tokens=args.max_tokens)
        inferred = tokenizer.decode(generated[0].tolist())
        print(inferred)
        print("-----------------------------------")


if __name__ == "__main__":
    main()
