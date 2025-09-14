from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, IterableDataset

from scratchgpt.data.hf_datasource import HFDataSource


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file for testing."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("Line 1\nLine 2\nLine 3\n")
    return text_file


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("text,label\nHello world,1\nTest data,0\n")
    return csv_file


@pytest.fixture
def temp_folder_with_texts(tmp_path: Path) -> Path:
    """Create a temporary folder with text files."""
    folder = tmp_path / "texts"
    folder.mkdir()
    (folder / "file1.txt").write_text("Content of file 1")
    (folder / "file2.txt").write_text("Content of file 2")
    return folder


def test_load_text_file(temp_text_file: Path):
    """Test loading a single text file."""
    ds = HFDataSource(temp_text_file)

    # Test iteration
    texts = list(ds)
    assert len(texts) == 3
    assert texts[0] == "Line 1"

    # Test length
    assert len(ds) == 3

    # Test indexing
    assert ds[1] == "Line 2"


def test_load_csv_file(temp_csv_file: Path):
    """Test loading a CSV file."""
    ds = HFDataSource(temp_csv_file, text_column="text")

    texts = list(ds)
    assert len(texts) == 2
    assert "Hello world" in texts[0]
    assert "Test data" in texts[1]


def test_load_folder(temp_folder_with_texts: Path):
    """Test loading a folder of text files."""
    ds = HFDataSource(temp_folder_with_texts)

    texts = list(ds)
    assert len(texts) >= 2

    # Check that content from both files is loaded
    all_text = " ".join(texts)
    assert "Content of file 1" in all_text
    assert "Content of file 2" in all_text


@patch("scratchgpt.data.hf_datasource.load_dataset")
def test_load_hub_dataset(mock_load_dataset: MagicMock):
    """Test loading a dataset from HuggingFace Hub."""
    # Mock the dataset
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__iter__ = lambda self: iter([{"text": "Sample 1"}, {"text": "Sample 2"}])
    mock_dataset.__len__ = lambda self: 2
    mock_dataset.__getitem__ = lambda self, idx: {"text": f"Sample {idx + 1}"}
    mock_load_dataset.return_value = mock_dataset

    ds = HFDataSource("wikitext-2-raw-v1")

    # Verify load_dataset was called correctly
    mock_load_dataset.assert_called_once_with(
        "wikitext-2-raw-v1",
        split="train",
        streaming=False,
    )

    # Test iteration
    texts = list(ds)
    assert len(texts) == 2
    assert texts[0] == "Sample 1"


@patch("scratchgpt.data.hf_datasource.load_dataset")
def test_map_function(mock_load_dataset: MagicMock):
    """Test the map function."""
    # Create a mock dataset with map method
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.map.return_value = mock_dataset
    mock_load_dataset.return_value = mock_dataset

    ds = HFDataSource("dummy")

    # Test map
    def dummy_tokenize(x):
        return {"tokens": [1, 2, 3]}

    mapped_ds = ds.map(dummy_tokenize, batched=True)

    # Verify map was called
    mock_dataset.map.assert_called_once()
    assert isinstance(mapped_ds, HFDataSource)


@patch("scratchgpt.data.hf_datasource.load_dataset")
def test_train_test_split(mock_load_dataset: MagicMock):
    """Test train/test splitting."""
    # Create mock dataset with train_test_split
    mock_dataset = MagicMock(spec=Dataset)
    mock_train = MagicMock(spec=Dataset)
    mock_test = MagicMock(spec=Dataset)
    mock_dataset.train_test_split.return_value = {
        "train": mock_train,
        "test": mock_test,
    }
    mock_load_dataset.return_value = mock_dataset

    ds = HFDataSource("dummy")

    # Test split
    splits = ds.train_test_split(test_size=0.2, seed=42)

    # Verify split was called
    mock_dataset.train_test_split.assert_called_once_with(
        test_size=0.2,
        train_size=None,
        seed=42,
    )

    assert "train" in splits
    assert "test" in splits
    assert isinstance(splits["train"], HFDataSource)
    assert isinstance(splits["test"], HFDataSource)


@patch("scratchgpt.data.hf_datasource.load_dataset")
def test_streaming_mode(mock_load_dataset: MagicMock):
    """Test that streaming mode raises appropriate errors for unsupported operations."""
    mock_dataset = MagicMock(spec=IterableDataset)
    mock_load_dataset.return_value = mock_dataset

    ds = HFDataSource("dummy", streaming=True)

    # These should raise TypeError for streaming datasets
    with pytest.raises(TypeError, match="Streaming datasets don't support len"):
        len(ds)

    with pytest.raises(TypeError, match="Streaming datasets don't support indexing"):
        ds[0]

    with pytest.raises(TypeError, match="Streaming datasets don't support train_test_split"):
        ds.train_test_split(test_size=0.2)


@patch("scratchgpt.data.hf_datasource.load_dataset")
def test_column_names_property(mock_load_dataset: MagicMock):
    """Test the column_names property."""
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.column_names = ["text", "label", "id"]
    mock_load_dataset.return_value = mock_dataset

    ds = HFDataSource("dummy")

    assert ds.column_names == ["text", "label", "id"]

    # Test when dataset has no column_names
    mock_dataset_no_cols = MagicMock(spec=Dataset)
    if hasattr(mock_dataset_no_cols, "column_names"):
        delattr(mock_dataset_no_cols, "column_names")
    mock_load_dataset.return_value = mock_dataset_no_cols

    ds2 = HFDataSource("dummy2")

    assert ds2.column_names is None
