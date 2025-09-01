import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from scratchgpt.dataloader import PretokenizedDataset
from scratchgpt.preprocess import (
    File2FileTokenizerPreprocessor,
    Folder2FileTokenizerPreprocessor,
    TokenizerPreprocessor,
)
from scratchgpt.tokenizer.base_tokenizer import Tokenizer


class MockTokenizer(Tokenizer):
    """A controlled tokenizer for predictable testing."""

    def __init__(self, vocab_size: int = 256):
        self._vocab_size = vocab_size
        self.mapping = {chr(ord("a") + i): i + 1 for i in range(26)}
        self.mapping[" "] = 27
        self.mapping["\n"] = 28
        self.mapping["€"] = 29

    def encode(self, text: str) -> list[int]:
        return [self.mapping.get(char, 0) for char in text]

    def decode(self, encoding: list[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocabulary(self) -> list[str]:
        raise NotImplementedError


class TestTokenizerPreprocessor(unittest.TestCase):
    def test_happy_case_tokenization(self):
        """Test standard tokenization with a simple string."""
        tokenizer = MockTokenizer()
        preprocessor = TokenizerPreprocessor(tokenizer)
        source = io.StringIO("ab c")
        sink = io.BytesIO()

        preprocessor(source, sink)

        sink.seek(0)
        result = np.frombuffer(sink.read(), dtype=preprocessor.dtype)
        expected = np.array([1, 2, 27, 3], dtype=preprocessor.dtype)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_selection(self):
        """Ensure correct numpy dtype is chosen based on vocab size."""
        # uint8
        preprocessor_small = TokenizerPreprocessor(MockTokenizer(vocab_size=255))
        self.assertEqual(preprocessor_small.dtype, np.uint8)

        # uint16
        preprocessor_medium = TokenizerPreprocessor(MockTokenizer(vocab_size=65535))
        self.assertEqual(preprocessor_medium.dtype, np.uint16)

        # uint32
        preprocessor_large = TokenizerPreprocessor(MockTokenizer(vocab_size=65536))
        self.assertEqual(preprocessor_large.dtype, np.uint32)

    def test_empty_input(self):
        """Test that an empty source results in an empty sink."""
        preprocessor = TokenizerPreprocessor(MockTokenizer())
        source = io.StringIO("")
        sink = io.BytesIO()

        preprocessor(source, sink)

        self.assertEqual(sink.getvalue(), b"")

    def test_chunking_and_multibyte_chars(self):
        """Ensure correct processing with small chunks and unicode."""
        preprocessor = TokenizerPreprocessor(MockTokenizer())
        text = "a€b"  # '€' is a multi-byte character
        source = io.StringIO(text)
        sink = io.BytesIO()

        # Chunk size of 1 character
        preprocessor(source, sink, chunk_size=1)

        sink.seek(0)
        result = np.frombuffer(sink.read(), dtype=preprocessor.dtype)
        expected = np.array([1, 29, 2], dtype=preprocessor.dtype)
        np.testing.assert_array_equal(result, expected)

    @patch("scratchgpt.preprocess.tqdm")
    def test_progress_bar_update(self, mock_tqdm):
        """Verify that the progress bar is updated."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar

        preprocessor = TokenizerPreprocessor(MockTokenizer())
        source = io.StringIO("abc")
        sink = io.BytesIO()

        preprocessor(source, sink, pbar=mock_pbar)

        # 'abc' is 3 bytes in utf-8
        mock_pbar.update.assert_called_once_with(3)


class TestFileAndFolderPreprocessors(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    # --- File2FileTokenizerPreprocessor Tests ---

    @patch("scratchgpt.preprocess.tqdm")
    def test_file2file_happy_case(self, mock_tqdm):
        """Test successful preprocessing of a single file."""
        tokenizer = MockTokenizer()
        preprocessor = File2FileTokenizerPreprocessor(tokenizer)

        input_file = self.test_path / "input.txt"
        output_file = self.test_path / "output.bin"
        input_file.write_text("a b c", encoding="utf-8")

        preprocessor(input_file, output_file)

        self.assertTrue(output_file.exists())
        result = np.fromfile(output_file, dtype=preprocessor._preprocessor.dtype)
        expected = np.array([1, 27, 2, 27, 3], dtype=preprocessor._preprocessor.dtype)
        np.testing.assert_array_equal(result, expected)

    def test_file2file_error_input_not_found(self):
        """Ensure error is raised if input file does not exist."""
        preprocessor = File2FileTokenizerPreprocessor(MockTokenizer())
        with self.assertRaises(ValueError):
            # The call to `is_file()` inside the preprocessor will fail
            preprocessor(self.test_path / "nonexistent.txt", self.test_path / "output.bin")

    def test_file2file_error_output_exists(self):
        """Ensure error is raised if output file already exists."""
        preprocessor = File2FileTokenizerPreprocessor(MockTokenizer())
        input_file = self.test_path / "input.txt"
        output_file = self.test_path / "output.bin"
        input_file.touch()
        output_file.touch()
        with self.assertRaises(FileExistsError):
            preprocessor(input_file, output_file)

    # --- Folder2FileTokenizerPreprocessor Tests ---

    @patch("scratchgpt.preprocess.tqdm")
    def test_folder2file_happy_case(self, mock_tqdm):
        """Test successful preprocessing of a directory."""
        preprocessor = Folder2FileTokenizerPreprocessor(MockTokenizer())

        # Setup directory structure
        (self.test_path / "sub").mkdir()
        (self.test_path / "file1.txt").write_text("a b", encoding="utf-8")
        (self.test_path / "file2.txt").write_text(" c d", encoding="utf-8")
        (self.test_path / "sub" / "file3.txt").write_text(" e", encoding="utf-8")
        # This file should be ignored
        (self.test_path / ".ignored.txt").touch()

        output_file = self.test_path / "output.bin"
        preprocessor(self.test_path, output_file)

        self.assertTrue(output_file.exists())
        result = np.fromfile(output_file, dtype=preprocessor._preprocessor.dtype)
        # Order is not guaranteed, so we sort both arrays
        result.sort()
        expected = np.array([1, 27, 2, 27, 3, 27, 4, 27, 5], dtype=preprocessor._preprocessor.dtype)
        expected.sort()
        np.testing.assert_array_equal(result, expected)

    def test_folder2file_error_input_is_file(self):
        """Ensure error is raised if input path is a file."""
        preprocessor = Folder2FileTokenizerPreprocessor(MockTokenizer())
        input_file = self.test_path / "input.txt"
        input_file.touch()
        with self.assertRaises(ValueError):
            preprocessor(input_file, self.test_path / "output.bin")

    def test_folder2file_empty_folder(self):
        """Test that an empty folder produces an empty output file."""
        preprocessor = Folder2FileTokenizerPreprocessor(MockTokenizer())
        output_file = self.test_path / "output.bin"
        preprocessor(self.test_path, output_file)
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.stat().st_size, 0)


class TestDatasetIntegration(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory and a predictable tokenizer."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        self.tokenizer = MockTokenizer(vocab_size=500)
        self.tokenizer.encode = lambda text: [int(x) for x in text.split()]

        # Common setup: create a preprocessed file with 100 tokens (0-99)
        self.block_size = 10
        self.num_tokens = 100
        self.token_file = self.test_path / "tokens.bin"
        preprocessor = File2FileTokenizerPreprocessor(self.tokenizer)
        input_text = " ".join(map(str, range(self.num_tokens)))
        input_file = self.test_path / "input.txt"
        input_file.write_text(input_text)
        preprocessor(input_file, self.token_file)

        self.dtype = np.dtype(np.uint16)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    def test_dataset_len_and_getitem(self):
        """Verify the full dataset's length and item retrieval."""
        dataset = PretokenizedDataset(self.token_file, self.block_size, dtype=self.dtype)

        # Check __len__
        expected_len = self.num_tokens - self.block_size
        self.assertEqual(len(dataset), expected_len)

        # Check __getitem__
        block, target = dataset[0]

        # Verify content
        expected_block = torch.arange(0, self.block_size, dtype=torch.int64)
        self.assertTrue(torch.equal(block, expected_block))

        # Verify that the dtype is converted to long (int64)
        self.assertEqual(block.dtype, torch.long)
        self.assertEqual(target.dtype, torch.long)

    def test_integration_with_random_split(self):
        """Verify the dataset works correctly with torch.utils.data.random_split."""
        from torch.utils.data import random_split

        full_dataset = PretokenizedDataset(self.token_file, self.block_size, dtype=self.dtype)

        # Use a generator for a deterministic split
        generator = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = random_split(full_dataset, [0.8, 0.1, 0.1], generator=generator)

        # Verify subset lengths (Note: random_split provides Subset objects)
        self.assertEqual(len(train_set), 72)
        self.assertEqual(len(val_set), 9)
        self.assertEqual(len(test_set), 9)

        # Check an item from a subset to ensure it proxies correctly
        block, target = train_set[0]  # Get the first item from the training Subset

        self.assertEqual(block.shape, (self.block_size,))
        self.assertEqual(target.shape, (self.block_size,))
        self.assertEqual(block.dtype, torch.long)

    def test_dataset_len_when_data_smaller_than_block_size(self):
        """Test the edge case where token count is less than block_size."""
        token_file = self.test_path / "small_tokens.bin"
        preprocessor = File2FileTokenizerPreprocessor(self.tokenizer)

        # Create a file with only 5 tokens
        input_text = " ".join(map(str, range(5)))
        input_file = self.test_path / "small_input.txt"
        input_file.write_text(input_text)
        preprocessor(input_file, token_file)

        # Use a block_size larger than the number of tokens
        dataset = PretokenizedDataset(token_file, block_size=10, dtype=np.dtype(np.uint16))

        # The length should be 0, not a negative number
        self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
