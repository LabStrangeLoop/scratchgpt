import os
import pickle

import torch

from scratchgpt.model.model import TransformerLanguageModel

from .tokenizer.base_tokenizer import Tokenizer
from .tokenizer.tiktoken import TiktokenWrapper


class ModelLoadFailedError(Exception):
    pass


def get_best_model_weights_path(exp_folder: str) -> str:
    return os.path.join(exp_folder, "best_model_weights.pth")


def get_latest_model_weights_path(exp_folder: str) -> str:
    return os.path.join(exp_folder, "latest_model_weights.pth")


def get_tokenizer_path(exp_folder: str) -> str:
    return os.path.join(exp_folder, "tokenizer.pkl")


def load_model(model_path: str, model: TransformerLanguageModel, device: torch.device) -> TransformerLanguageModel:
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


def get_tokenizer(exp_path: str) -> Tokenizer:
    tokenizer_path = get_tokenizer_path(exp_path)
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer: Tokenizer = pickle.load(f)
    else:
        tokenizer = TiktokenWrapper("cl100k_base")
    return tokenizer


def save_tokenizer(exp_path: str, tokenizer: Tokenizer) -> None:
    tokenizer_path = get_tokenizer_path(exp_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
        print(f"Saved the tokenizer to path: {tokenizer_path}")
