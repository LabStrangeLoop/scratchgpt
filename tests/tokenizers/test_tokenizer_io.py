from pathlib import Path

import pytest

from scratchgpt.model_io import (
    TokenizerLoadFailedError,
    load_tokenizer,
    save_tokenizer,
)
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer

# A simple corpus for creating tokenizers in tests
TEST_CORPUS = "hello world"


def test_load_tokenizer_loads_existing(
    tmp_path: Path,
) -> None:
    """
    Tests that `load_tokenizer` correctly loads an existing tokenizer
    from a path and ignores the default factory.
    """
    # Setup: Create and save a tokenizer to the temp directory first
    initial_tokenizer = CharTokenizer(text="abcde")
    save_tokenizer(tmp_path, initial_tokenizer)

    # Action: Call load_tokenizer on the populated directory.
    # The factory now uses a different corpus to ensure it's not being called.
    loaded_tokenizer = load_tokenizer(exp_path=tmp_path)

    # Assertions
    assert isinstance(loaded_tokenizer, CharTokenizer)
    # The vocab size should match the *saved* tokenizer ("abcde"), not the factory one.
    assert loaded_tokenizer.vocab_size == 5
    assert loaded_tokenizer.decode([0, 1, 2]) == "abc"


def test_load_tokenizer_raises_on_bad_config_type(tmp_path: Path) -> None:
    """
    Tests that `load_tokenizer` raises an error if the config file
    points to an unregistered tokenizer type.
    """
    # Setup: Manually create a bad tokenizer config file
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    bad_config = '{"tokenizer_type": "UnregisteredTokenizer"}'
    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        f.write(bad_config)

    # Action & Assertion: Expect a TokenizerLoadFailedError
    with pytest.raises(TokenizerLoadFailedError, match="Unknown tokenizer type"):
        load_tokenizer(exp_path=tmp_path)


def test_get_tokenizer_raises_on_missing_config_field(tmp_path: Path) -> None:
    """
    Tests that `load_tokenizer` raises an error if the tokenizer
    config file is missing the 'tokenizer_type' field.
    """
    # Setup: Manually create a malformed tokenizer config file
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    bad_config = '{"some_other_field": "some_value"}'
    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        f.write(bad_config)

    # Action & Assertion: Expect a TokenizerLoadFailedError
    with pytest.raises(TokenizerLoadFailedError, match="missing 'tokenizer_type' field"):
        load_tokenizer(exp_path=tmp_path)
