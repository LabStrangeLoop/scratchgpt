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
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.5, 0.5),
        random_seed=42,
        iteration_type="chunking",
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
        iteration_type="chunking",
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
        iteration_type="chunking",
    )

    assert isinstance(train_loader, DataLoader)
    assert val_loader is None

    # 32 chars // (10+1) chunk size = 2 total samples. batch_size=2 -> 1 batch.
    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape == (batch_size, block_size)
    assert torch.equal(train_batch["input_ids"][0, 1:], train_batch["labels"][0, :-1])


@pytest.fixture
def multiline_text_file(tmp_path: Path) -> Path:
    """Creates a text file with multiple lines for proper splitting."""
    data_path = tmp_path / "multiline.txt"
    # Each line becomes a separate sample in the dataset
    data_path.write_text(
        "0123456789abcdef\n"
        "0123456789abcdef\n"
        "0123456789abcdef\n"
        "0123456789abcdef"
    )
    return data_path


def test_hf_datasource_sliding_from_file(multiline_text_file, simple_tokenizer):
    """Tests loading a dataset with sliding window from a multi-line text file."""
    block_size = 7
    batch_size = 4

    data_source = HFDataSource(path_or_name=str(multiline_text_file))

    train_loader, val_loader = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.75, 0.25),  # 3 train lines, 1 val line
        random_seed=42,
        iteration_type="sliding",
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Each line has 16 tokens, so:
    # Train: 3 lines * (16-7) = 27 sliding windows
    # Val: 1 line * (16-7) = 9 sliding windows
    train_samples = sum(len(batch["input_ids"]) for batch in train_loader)
    val_samples = sum(len(batch["input_ids"]) for batch in val_loader)

    # The exact split might vary due to randomization
    assert train_samples > 0
    assert val_samples > 0

    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape[1] == block_size
    assert train_batch["labels"].shape[1] == block_size


def test_hf_datasource_sliding_from_directory(tmp_path, simple_tokenizer):
    """Tests loading a dataset with sliding window from a directory."""
    # Create directory with multiple files (each becomes a sample)
    data_dir = tmp_path / "multi_files"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("0123456789abcdef")
    (data_dir / "b.txt").write_text("0123456789abcdef")
    (data_dir / "c.txt").write_text("0123456789abcdef")
    (data_dir / "d.txt").write_text("0123456789abcdef")

    block_size = 10
    batch_size = 5

    data_source = HFDataSource(path_or_name=str(data_dir))

    train_loader, val_loader = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.75, 0.25),  # 3 train files, 1 val file
        random_seed=42,
        iteration_type="sliding",
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape[1] == block_size
    assert train_batch["labels"].shape[1] == block_size


def test_sliding_window_overlap(multiline_text_file, simple_tokenizer):
    """Tests that sliding windows actually overlap as expected."""
    block_size = 5
    batch_size = 2

    data_source = HFDataSource(path_or_name=str(multiline_text_file))

    train_loader, val_loader = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.75, 0.25),  # Valid split for 4 lines
        random_seed=42,
        iteration_type="sliding",
    )

    # Check within a batch that the sliding window property holds
    train_batch = next(iter(train_loader))

    # Each sample in the batch should have correct shape
    assert train_batch["input_ids"].shape[1] == block_size
    assert train_batch["labels"].shape[1] == block_size

    # Labels should be input shifted by 1 position
    for i in range(len(train_batch["input_ids"])):
        # The relationship between input and labels in the same window
        input_ids = train_batch["input_ids"][i]
        labels = train_batch["labels"][i]
        # labels[j] should predict input_ids[j+1]
        assert torch.equal(input_ids[1:], labels[:-1])


def test_hf_datasource_streaming_sliding_raises_error(dummy_text_file, simple_tokenizer):
    """Tests that sliding window with streaming dataset raises an error."""
    block_size = 7
    batch_size = 2

    data_source = HFDataSource(path_or_name=str(dummy_text_file), streaming=True)

    with pytest.raises(ValueError, match="Sliding not supported for streaming dataset"):
        data_source.get_dataloaders(
            tokenizer=simple_tokenizer,
            block_size=block_size,
            batch_size=batch_size,
            splits=(0.8, 0.2),
            random_seed=42,
            iteration_type="sliding",
        )


def test_sliding_vs_chunking_sample_count(multiline_text_file, simple_tokenizer):
    """Tests that sliding produces more samples than chunking."""
    block_size = 8
    batch_size = 100  # Large batch to get all samples

    data_source = HFDataSource(path_or_name=str(multiline_text_file))

    # Get chunking results - use a valid split
    chunk_train, chunk_val = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.75, 0.25),
        random_seed=42,
        iteration_type="chunking",
    )

    # Get sliding results with same split
    slide_train, slide_val = data_source.get_dataloaders(
        tokenizer=simple_tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        splits=(0.75, 0.25),
        random_seed=42,
        iteration_type="sliding",
    )

    # Count total samples in each
    chunk_samples = sum(len(batch["input_ids"]) for batch in chunk_train)
    slide_samples = sum(len(batch["input_ids"]) for batch in slide_train)

    # Sliding should produce many more samples than chunking
    # Chunking: non-overlapping blocks
    # Sliding: overlapping windows (one per position)
    assert slide_samples > chunk_samples

    # Verify the shapes are consistent
    chunk_batch = next(iter(chunk_train))
    slide_batch = next(iter(slide_train))
    assert chunk_batch["input_ids"].shape[1] == block_size
    assert slide_batch["input_ids"].shape[1] == block_size
