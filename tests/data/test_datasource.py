from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scratchgpt.data.hf_datasource import HFDataSource
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer


@pytest.fixture
def simple_tokenizer() -> CharTokenizer:
    """A simple tokenizer where token ID matches the digit."""
    return CharTokenizer(text="0123456789abcdef")


@pytest.fixture
def dummy_text_file(tmp_path: Path) -> Path:
    """Creates a dummy text file with 32 chars."""
    data_path = tmp_path / "data.txt"
    data_path.write_text("0123456789abcdef0123456789abcdef")
    return data_path


@pytest.fixture
def dummy_text_dir(tmp_path: Path) -> Path:
    """Creates a directory with two text files for a total of 32 chars."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("0123456789abcdef")
    (data_dir / "b.txt").write_text("0123456789abcdef")
    return data_dir


def test_hf_datasource_from_file(dummy_text_file, simple_tokenizer):
    """Tests loading a standard dataset from a single .txt file."""
    block_size = 7
    batch_size = 2

    data_source = HFDataSource(path_or_name=str(dummy_text_file))

    train_loader, val_loader = data_source.get_dataloaders(
        tokenizer=simple_tokenizer, block_size=block_size, batch_size=batch_size, splits=(0.5, 0.5), random_seed=42
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # 32 chars // (7+1) chunk size = 4 total samples. Split 50/50 -> 2 train, 2 val.
    # With batch_size=2, there should be 1 batch in each loader.
    assert len(train_loader) == 1
    assert len(val_loader) == 1

    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape == (batch_size, block_size)
    assert torch.equal(train_batch["input_ids"][0, 1:], train_batch["labels"][0, :-1])


@pytest.mark.skip
def test_hf_datasource_from_directory(dummy_text_dir, simple_tokenizer):
    """Tests loading a standard dataset from a directory of text files."""
    block_size = 7
    batch_size = 4  # Use all data in one batch

    data_source = HFDataSource(path_or_name=str(dummy_text_dir))

    train_loader, _ = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(1.0, 0.0),  # Use all data for training
        random_seed=42,
    )

    # 32 chars // (7+1) chunk size = 4 total samples.
    assert len(train_loader) == 1
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (batch_size, block_size)


def test_hf_datasource_streaming_from_file(dummy_text_file, simple_tokenizer):
    """Tests loading a streaming dataset from a single .txt file."""
    block_size = 10
    batch_size = 2

    data_source = HFDataSource(path_or_name=str(dummy_text_file), streaming=True)

    train_loader, val_loader = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.8, 0.2),  # Splits are ignored for streaming
        random_seed=42,
    )

    assert isinstance(train_loader, DataLoader)
    assert val_loader is None

    # 32 chars // (10+1) chunk size = 2 total samples. batch_size=2 -> 1 batch.
    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape == (batch_size, block_size)
    assert torch.equal(train_batch["input_ids"][0, 1:], train_batch["labels"][0, :-1])
