"""
ScratchGPT: A small-scale transformer-based language model implemented from scratch.
"""

from scratchgpt.config import (
    ScratchGPTArchitecture,
    ScratchGPTConfig,
    ScratchGPTTraining,
)
from scratchgpt.data.datasource import (
    ByteSizableDataSource,
    DataSource,
    FileDataSource,
    FolderDataSource,
    LineByLineFileDataSource,
)
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.model_io import (
    ModelLoadFailedError,
    TokenizerLoadFailedError,
    get_best_model_weights_path,
    get_latest_model_weights_path,
    get_tokenizer_path,
    load_model,
    load_tokenizer,
    save_tokenizer,
)
from scratchgpt.tokenizer.base_tokenizer import (
    SerializableTokenizer,
    Tokenizer,
    register_tokenizer,
)
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer, Utf8Tokenizer
from scratchgpt.tokenizer.hf_tokenizer import HuggingFaceTokenizer
from scratchgpt.training.trainer import Trainer, get_dtype_for_vocab_size

__all__ = [
    # Core Model and Config
    "TransformerLanguageModel",
    "ScratchGPTConfig",
    "ScratchGPTArchitecture",
    "ScratchGPTTraining",
    # Data Sources
    "DataSource",
    "ByteSizableDataSource",
    "FileDataSource",
    "FolderDataSource",
    "LineByLineFileDataSource",
    # Model I/O
    "load_model",
    "load_tokenizer",
    "save_tokenizer",
    "get_best_model_weights_path",
    "get_latest_model_weights_path",
    "get_tokenizer_path",
    "ModelLoadFailedError",
    "TokenizerLoadFailedError",
    # Tokenizers
    "Tokenizer",
    "SerializableTokenizer",
    "register_tokenizer",
    "CharTokenizer",
    "Utf8Tokenizer",
    "HuggingFaceTokenizer",
    # Training
    "Trainer",
    "get_dtype_for_vocab_size",
]
