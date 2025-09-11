from collections.abc import Callable
from pathlib import Path

import pytest

from scratchgpt.model_io import (
    TokenizerLoadFailedError,
    get_tokenizer,
    save_tokenizer,
)
from scratchgpt.tokenizer.base_tokenizer import SerializableTokenizer
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer

# A simple corpus for creating tokenizers in tests
TEST_CORPUS = "hello world"


@pytest.fixture
def char_tokenizer_factory() -> Callable[[], SerializableTokenizer]:
    """Provides a factory to create a simple CharTokenizer for tests."""
    return lambda: CharTokenizer(text=TEST_CORPUS)


def test_get_tokenizer_creates_new_from_factory(
    tmp_path: Path, char_tokenizer_factory: Callable[[], SerializableTokenizer]
):
    """
    Tests that `get_tokenizer` correctly creates a new tokenizer
    using the factory when no tokenizer exists at the path.
    """
    # Action: Call get_tokenizer on an empty directory
    tokenizer = get_tokenizer(exp_path=tmp_path, default_factory=char_tokenizer_factory)

    # Assertions
    assert isinstance(tokenizer, CharTokenizer)
    assert tokenizer.vocab_size == len(set(TEST_CORPUS))
    # The function itself doesn't save, so the path should still be empty
    tokenizer_config_path = tmp_path / "tokenizer" / "tokenizer_config.json"
    assert not tokenizer_config_path.exists()


def test_get_tokenizer_loads_existing(tmp_path: Path, char_tokenizer_factory: Callable[[], SerializableTokenizer]):
    """
    Tests that `get_tokenizer` correctly loads an existing tokenizer
    from a path and ignores the default factory.
    """
    # Setup: Create and save a tokenizer to the temp directory first
    initial_tokenizer = CharTokenizer(text="abcde")
    save_tokenizer(tmp_path, initial_tokenizer)

    # Action: Call get_tokenizer on the populated directory.
    # The factory now uses a different corpus to ensure it's not being called.
    loaded_tokenizer = get_tokenizer(exp_path=tmp_path, default_factory=char_tokenizer_factory)

    # Assertions
    assert isinstance(loaded_tokenizer, CharTokenizer)
    # The vocab size should match the *saved* tokenizer ("abcde"), not the factory one.
    assert loaded_tokenizer.vocab_size == 5
    assert loaded_tokenizer.decode([0, 1, 2]) == "abc"


def test_get_tokenizer_raises_on_bad_config_type(tmp_path: Path):
    """
    Tests that `get_tokenizer` raises an error if the config file
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
        get_tokenizer(exp_path=tmp_path, default_factory=lambda: None)


def test_get_tokenizer_raises_on_missing_config_field(tmp_path: Path):
    """
    Tests that `get_tokenizer` raises an error if the tokenizer
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
        get_tokenizer(exp_path=tmp_path, default_factory=lambda: None)
