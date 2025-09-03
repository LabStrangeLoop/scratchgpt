import json
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

try:
    from tokenizers import Tokenizer as HFTokenizer

    from scratchgpt.tokenizer.hf_tokenizer import HuggingFaceTokenizer

    hf_tokenizers_installed = True
except ImportError:
    hf_tokenizers_installed = False

from scratchgpt.model_io import get_tokenizer, save_tokenizer
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer
from scratchgpt.tokenizer.tiktoken import TiktokenWrapper


@pytest.fixture
def temp_experiment_dir(tmp_path: Path) -> Generator[Path, Any, Any]:
    """Pytest fixture to create a temporary directory for each test."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    yield exp_dir
    # Teardown is handled by tmp_path fixture
    shutil.rmtree(exp_dir, ignore_errors=True)


# --- Tests for CharTokenizer ---


class TestCharTokenizerIO:
    def test_save_and_load_happy_path(self, temp_experiment_dir: Path) -> None:
        """Tests standard saving and loading of a CharTokenizer."""
        original_text = "hello world"
        original_tokenizer = CharTokenizer(text=original_text)

        save_tokenizer(temp_experiment_dir, original_tokenizer)
        loaded_tokenizer = get_tokenizer(temp_experiment_dir)

        assert isinstance(loaded_tokenizer, CharTokenizer)
        assert loaded_tokenizer.vocabulary == original_tokenizer.vocabulary
        assert loaded_tokenizer.decode(loaded_tokenizer.encode("hello")) == "hello"

    def test_save_and_load_edge_cases(self, temp_experiment_dir: Path) -> None:
        """Tests edge cases like empty and unicode characters."""
        # Empty text
        empty_tokenizer = CharTokenizer(text="")
        save_tokenizer(temp_experiment_dir, empty_tokenizer)
        loaded_empty = get_tokenizer(temp_experiment_dir)
        assert isinstance(loaded_empty, CharTokenizer)
        assert loaded_empty.vocabulary == []

        # Unicode characters
        shutil.rmtree(temp_experiment_dir / "tokenizer", ignore_errors=True)
        unicode_text = "你好世界-नमस्ते दुनिया-こんにちは世界"
        unicode_tokenizer = CharTokenizer(text=unicode_text)
        save_tokenizer(temp_experiment_dir, unicode_tokenizer)
        loaded_unicode = get_tokenizer(temp_experiment_dir)
        assert isinstance(loaded_unicode, CharTokenizer)
        assert sorted(loaded_unicode.vocabulary) == sorted(set(unicode_text))

    def test_load_error_missing_vocab_file(self, temp_experiment_dir: Path) -> None:
        """Tests that loading fails if vocab.json is missing."""
        tokenizer_dir = temp_experiment_dir / "tokenizer"
        tokenizer_dir.mkdir()
        config = {"tokenizer_type": "CharTokenizer"}
        with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(FileNotFoundError, match="Vocabulary file not found"):
            get_tokenizer(temp_experiment_dir)


# --- Tests for HuggingFaceTokenizer ---


@pytest.mark.skipif(not hf_tokenizers_installed, reason="hf-tokenizers optional dependency not installed")
class TestHuggingFaceTokenizerIO:
    @pytest.fixture
    def gpt2_hf_tokenizer(self) -> HFTokenizer:
        """Fixture to create a mock/simple HF tokenizer instance."""
        # Create a simple BPE tokenizer in memory to avoid network calls in tests
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        hf_tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
        hf_tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], vocab_size=1000)
        hf_tokenizer.train_from_iterator(["This is a test sentence for gpt2 tokenizer"], trainer=trainer)
        return hf_tokenizer

    def test_save_and_load_happy_path(self, temp_experiment_dir: Path, gpt2_hf_tokenizer: HFTokenizer) -> None:
        """Tests standard saving and loading of a HuggingFaceTokenizer."""
        original_tokenizer = HuggingFaceTokenizer(tokenizer=gpt2_hf_tokenizer)

        save_tokenizer(temp_experiment_dir, original_tokenizer)
        loaded_tokenizer = get_tokenizer(temp_experiment_dir)

        assert isinstance(loaded_tokenizer, HuggingFaceTokenizer)
        assert loaded_tokenizer.vocab_size == original_tokenizer.vocab_size
        test_text = "This is a test"
        assert loaded_tokenizer.decode(loaded_tokenizer.encode(test_text)) == test_text

    def test_load_error_missing_tokenizer_json(self, temp_experiment_dir: Path) -> None:
        """Tests that loading fails if tokenizer.json is missing."""
        tokenizer_dir = temp_experiment_dir / "tokenizer"
        tokenizer_dir.mkdir()
        config = {"tokenizer_type": "HuggingFaceTokenizer"}
        with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(FileNotFoundError, match="Hugging Face tokenizer file not found"):
            get_tokenizer(temp_experiment_dir)

    @patch("scratchgpt.tokenizer.hf_tokenizer.hf_hub_download")
    def test_from_hub_mocked(
        self, mock_hub_download: MagicMock, temp_experiment_dir: Path, gpt2_hf_tokenizer: HFTokenizer
    ) -> None:
        """Tests loading from hub is correctly mocked."""
        # Save a temporary tokenizer file to simulate downloading
        local_path = temp_experiment_dir / "mock_tokenizer.json"
        gpt2_hf_tokenizer.save(str(local_path))
        mock_hub_download.return_value = str(local_path)

        tokenizer = HuggingFaceTokenizer.from_hub(repo_id="gpt2-mock")

        mock_hub_download.assert_called_once_with(repo_id="gpt2-mock", filename="tokenizer.json")
        assert isinstance(tokenizer, HuggingFaceTokenizer)
        assert tokenizer.vocab_size > 0


# --- Tests for Generic I/O Logic ---


class TestGenericIO:
    def test_get_tokenizer_default_fallback(self, temp_experiment_dir: Path) -> None:
        """Tests that get_tokenizer falls back to Tiktoken if no tokenizer is saved."""
        tokenizer = get_tokenizer(temp_experiment_dir)
        assert isinstance(tokenizer, TiktokenWrapper)

    def test_save_unserializable_tokenizer(self, temp_experiment_dir: Path) -> None:
        """Tests that saving a non-serializable tokenizer does nothing gracefully."""
        tokenizer = TiktokenWrapper()
        save_tokenizer(temp_experiment_dir, tokenizer)
        # The main assertion is that no directory is created and no error is raised
        assert not (temp_experiment_dir / "tokenizer").exists()

    def test_load_error_missing_config_key(self, temp_experiment_dir: Path) -> None:
        """Tests failure when tokenizer_type key is missing from config."""
        tokenizer_dir = temp_experiment_dir / "tokenizer"
        tokenizer_dir.mkdir()
        config = {"some_other_key": "some_value"}
        with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="Tokenizer config is missing 'tokenizer_type' field."):
            get_tokenizer(temp_experiment_dir)

    def test_load_error_unknown_tokenizer_type(self, temp_experiment_dir: Path) -> None:
        """Tests failure when tokenizer_type is not in the registry."""
        tokenizer_dir = temp_experiment_dir / "tokenizer"
        tokenizer_dir.mkdir()
        config = {"tokenizer_type": "MyImaginaryTokenizer"}
        with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="Unknown tokenizer type 'MyImaginaryTokenizer'"):
            get_tokenizer(temp_experiment_dir)
