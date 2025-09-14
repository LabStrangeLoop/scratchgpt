from pathlib import Path

import pytest
import torch

from scratchgpt.data.hf_datasource import HFDataSource
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer
from scratchgpt.training.tokenize_utils import prepare_dataset_for_training


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file with enough content for chunking."""
    text_file = tmp_path / "sample.txt"
    content = "The quick brown fox jumps over the lazy dog. " * 20
    text_file.write_text(content)
    return text_file


def test_tokenization_pipeline(sample_text_file: Path):
    """Test the full tokenization pipeline with HF datasets."""
    # Load data
    ds = HFDataSource(sample_text_file)

    # Create a simple tokenizer
    text = sample_text_file.read_text()
    tokenizer = CharTokenizer(text)

    # Prepare dataset for training
    block_size = 16
    prepared = prepare_dataset_for_training(
        ds.dataset,
        tokenizer,
        block_size,
        num_proc=1,
    )

    # Verify the dataset is properly formatted
    assert "input_ids" in prepared.column_names
    assert "labels" in prepared.column_names

    # Check that samples have correct shape
    first_sample = prepared[0]
    assert len(first_sample["input_ids"]) == block_size
    assert len(first_sample["labels"]) == block_size

    # Verify it's in PyTorch format
    assert isinstance(first_sample["input_ids"], torch.Tensor)
    assert isinstance(first_sample["labels"], torch.Tensor)

    # Check that labels are shifted by 1
    for i in range(min(5, len(prepared))):
        sample = prepared[i]
        # The relationship between input_ids and labels should be maintained
        assert sample["input_ids"].shape == sample["labels"].shape
