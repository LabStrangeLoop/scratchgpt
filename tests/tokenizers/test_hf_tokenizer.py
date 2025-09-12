from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Attempt to import optional dependencies
try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    from scratchgpt.tokenizer.hf_tokenizer import HuggingFaceTokenizer

    hf_tokenizers_installed = True
except ImportError:
    hf_tokenizers_installed = False

from scratchgpt.model_io import save_tokenizer

# Skip all tests in this file if the optional dependencies are not installed
pytestmark = pytest.mark.skipif(
    not hf_tokenizers_installed,
    reason="hf-tokenizers optional dependency not installed",
)


@pytest.fixture
def simple_hf_tokenizer() -> HFTokenizer:
    """Fixture to create a simple BPE tokenizer in memory for tests."""
    hf_tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    hf_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], vocab_size=100)
    training_corpus = ["A test sentence for the tokenizer", "This is a test"]
    hf_tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    return hf_tokenizer


def test_save_and_load_happy_path(tmp_path: Path, simple_hf_tokenizer: HFTokenizer):
    """Tests standard saving and loading of a HuggingFaceTokenizer."""
    original_tokenizer = HuggingFaceTokenizer(tokenizer=simple_hf_tokenizer)
    tokenizer_dir = tmp_path / "experiment"

    save_tokenizer(tokenizer_dir, original_tokenizer)

    # Directly test the class's .load() method
    loaded_tokenizer = HuggingFaceTokenizer.load(tokenizer_dir / "tokenizer")

    assert isinstance(loaded_tokenizer, HuggingFaceTokenizer)
    assert loaded_tokenizer.vocab_size == original_tokenizer.vocab_size
    test_text = "This is a test"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(test_text)) == test_text


def test_load_error_missing_tokenizer_json(tmp_path: Path):
    """Tests that HuggingFaceTokenizer.load() fails if tokenizer.json is missing."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Hugging Face tokenizer file not found"):
        HuggingFaceTokenizer.load(tokenizer_dir)


@patch("scratchgpt.tokenizer.hf_tokenizer.hf_hub_download")
def test_from_hub_mocked(mock_hub_download: MagicMock, tmp_path: Path, simple_hf_tokenizer: HFTokenizer):
    """Tests that the .from_hub() classmethod correctly calls the download utility."""
    # Save a temporary tokenizer file to simulate it being downloaded
    local_path = tmp_path / "mock_tokenizer.json"
    simple_hf_tokenizer.save(str(local_path))
    mock_hub_download.return_value = str(local_path)

    tokenizer = HuggingFaceTokenizer.from_hub(repo_id="gpt2-mock")

    mock_hub_download.assert_called_once_with(repo_id="gpt2-mock", filename="tokenizer.json")
    assert isinstance(tokenizer, HuggingFaceTokenizer)
    assert tokenizer.vocab_size > 0
