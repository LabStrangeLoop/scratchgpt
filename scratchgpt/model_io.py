import json
import os
from pathlib import Path

import torch

from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer import char_tokenizer, hf_tokenizer  # noqa
from scratchgpt.tokenizer.base_tokenizer import TOKENIZER_REGISTRY, SerializableTokenizer, Tokenizer


class ModelLoadFailedError(Exception):
    """Raised when model loading fails"""


class TokenizerLoadFailedError(Exception):
    """Raised when a tokenizer cannot be loaded from a directory."""


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


def load_tokenizer(exp_path: Path) -> SerializableTokenizer:
    """
    Loads a saved tokenizer from an experiment directory.

    This function is intended for inference, where a tokenizer must already
    exist. It will raise an error if no tokenizer is found.
    """
    tokenizer_dir = exp_path / "tokenizer"
    config_path = tokenizer_dir / "tokenizer_config.json"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer config not found at '{config_path}'. "
            "Ensure the model has been trained and a tokenizer was saved."
        )

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_type = config.get("tokenizer_type")
    if not tokenizer_type:
        raise TokenizerLoadFailedError("Tokenizer config is missing 'tokenizer_type' field.")

    tokenizer_class = TOKENIZER_REGISTRY.get(tokenizer_type)
    if not tokenizer_class:
        raise TokenizerLoadFailedError(
            f"Unknown tokenizer type '{tokenizer_type}' in config. Ensure it's registered with @register_tokenizer."
        )

    print(f"✅ Loading tokenizer of type '{tokenizer_type}'...")
    return tokenizer_class.load(tokenizer_dir)


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
