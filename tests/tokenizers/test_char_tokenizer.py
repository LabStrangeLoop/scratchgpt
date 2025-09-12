import json
from pathlib import Path

import pytest

from scratchgpt.model_io import save_tokenizer
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer


def test_save_and_load_happy_path(tmp_path: Path):
    """Tests standard saving and loading of a CharTokenizer."""
    original_text = "hello world"
    original_tokenizer = CharTokenizer(text=original_text)
    tokenizer_dir = tmp_path / "experiment"

    save_tokenizer(tokenizer_dir, original_tokenizer)

    # Use the class's own .load() method for a direct unit test
    loaded_tokenizer = CharTokenizer.load(tokenizer_dir / "tokenizer")

    assert isinstance(loaded_tokenizer, CharTokenizer)
    assert loaded_tokenizer.vocabulary == original_tokenizer.vocabulary
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("hello")) == "hello"


def test_save_and_load_edge_cases(tmp_path: Path):
    """Tests edge cases like empty and unicode characters."""
    # --- Empty text ---
    empty_tokenizer = CharTokenizer(text="")
    empty_dir = tmp_path / "empty_exp"
    save_tokenizer(empty_dir, empty_tokenizer)
    loaded_empty = CharTokenizer.load(empty_dir / "tokenizer")

    assert isinstance(loaded_empty, CharTokenizer)
    assert loaded_empty.vocabulary == []

    # --- Unicode characters ---
    unicode_text = "你好世界-नमस्ते दुनिया-こんにちは世界"
    unicode_tokenizer = CharTokenizer(text=unicode_text)
    unicode_dir = tmp_path / "unicode_exp"
    save_tokenizer(unicode_dir, unicode_tokenizer)
    loaded_unicode = CharTokenizer.load(unicode_dir / "tokenizer")

    assert isinstance(loaded_unicode, CharTokenizer)
    assert sorted(loaded_unicode.vocabulary) == sorted(set(unicode_text))


def test_load_error_missing_vocab_file(tmp_path: Path):
    """Tests that CharTokenizer.load() fails if vocab.json is missing."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    # Manually create only the config file, but not the vocab file
    config = {"tokenizer_type": "CharTokenizer", "vocab_file": "vocab.json"}
    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f)

    with pytest.raises(FileNotFoundError, match="Vocabulary file not found"):
        CharTokenizer.load(tokenizer_dir)
