import json
import os
from pathlib import Path

import torch

from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer import char_tokenizer, hf_tokenizer  # noqa
from scratchgpt.tokenizer.base_tokenizer import TOKENIZER_REGISTRY, SerializableTokenizer, Tokenizer
from scratchgpt.tokenizer.tiktoken import TiktokenWrapper


class ModelLoadFailedError(Exception):
    pass


def get_best_model_weights_path(exp_folder: Path) -> Path:
    return exp_folder / "best_model_weights.pth"


def get_latest_model_weights_path(exp_folder: Path) -> Path:
    return exp_folder / "latest_model_weights.pth"


def get_tokenizer_path(exp_folder: Path) -> Path:
    return exp_folder / "tokenizer"


def load_model(model_path: Path, model: TransformerLanguageModel, device: torch.device) -> TransformerLanguageModel:
    model.to(device)
    if os.path.exists(model_path):
        try:
            print(f"Loading weights from: {model_path}")
            model_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(model_dict)
        except Exception as error:
            raise ModelLoadFailedError(model_path) from error
    else:
        print("No saved model, starting from scratch...gpt, lol!")
    return model


def get_tokenizer(exp_path: Path) -> Tokenizer:
    """
    Loads a tokenizer from the experiment directory.

    This function reads the `tokenizer_config.json` to determine the correct
    tokenizer type and then uses its `load` method. If no saved tokenizer
    is found, it defaults to Tiktoken.
    """
    tokenizer_dir = get_tokenizer_path(exp_path)
    config_path = tokenizer_dir / "tokenizer_config.json"

    if config_path.is_file():
        print(f"Found tokenizer config at: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        tokenizer_type = config.get("tokenizer_type")
        if not tokenizer_type:
            raise ValueError("Tokenizer config is missing 'tokenizer_type' field.")

        tokenizer_class = TOKENIZER_REGISTRY.get(tokenizer_type)

        if tokenizer_class:
            print(f"Loading tokenizer of type '{tokenizer_type}'...")
            return tokenizer_class.load(tokenizer_dir)
        else:
            raise ValueError(f"Unknown tokenizer type '{tokenizer_type}' in config.")

    else:
        print("No saved tokenizer found. Defaulting to Tiktoken 'cl100k_base'.")
        return TiktokenWrapper("cl100k_base")


def save_tokenizer(exp_path: Path, tokenizer: Tokenizer) -> None:
    """
    Saves a tokenizer if it supports the SerializableTokenizer interface.
    """
    if isinstance(tokenizer, SerializableTokenizer):
        tokenizer_path = get_tokenizer_path(exp_path)
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to path: {tokenizer_path}")
    else:
        print(f"Tokenizer of type '{type(tokenizer).__name__}' is not serializable and will not be saved.")
