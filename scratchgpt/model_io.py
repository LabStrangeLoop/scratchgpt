import os
import pickle

import torch
from torch import nn

from scratchgpt.tokenizer.neat_tokenizer import NeatGenomeTokenizer

from .tokenizer.base_tokenizer import Tokenizer
from .tokenizer.tiktoken import TiktokenWrapper


class ModelLoadFailed(Exception):
    pass


def get_best_model_weights_path(exp_folder: str) -> str:
    return os.path.join(exp_folder, "best_model_weights.pth")


def get_latest_model_weights_path(exp_folder: str) -> str:
    return os.path.join(exp_folder, "latest_model_weights.pth")


def get_tokenizer_path(exp_folder: str) -> str:
    return os.path.join(exp_folder, "tokenizer.pkl")


def load_model(model_path: str, model: nn.Module, device: torch.device) -> None:
    if os.path.exists(model_path):
        try:
            print(f"Loading weights from: {model_path}")
            model_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(model_dict)
            model.to(device)
        except Exception:
            raise ModelLoadFailed(model_path)
    else:
        print("No model path exists, proceeding with a new model")


def get_tokenizer(exp_path: str, default_tokenizer: str) -> Tokenizer:
    tokenizer_path = get_tokenizer_path(exp_path)
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer: Tokenizer = pickle.load(f)
    else:
        match default_tokenizer:
            case "neat":
                # TODO: Aleks likes magic numbers
                tokenizer = NeatGenomeTokenizer(-3.0, 3.0, 2)
            case _:
                tokenizer = TiktokenWrapper("cl100k_base")
    return tokenizer


def save_tokenizer(exp_path: str, tokenizer: Tokenizer) -> None:
    tokenizer_path = get_tokenizer_path(exp_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
        print(f"Saved the tokenizer to path: {tokenizer_path}")
