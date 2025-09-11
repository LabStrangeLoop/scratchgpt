from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from tqdm.auto import tqdm


@runtime_checkable
class DataSource(Protocol):
    """
    An interface for providing raw data to the Trainer.

    A DataSource is an iterable object that yields individual,
    untokenized training samples as strings.
    """

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the raw text samples."""
        ...


class FileDataSource(DataSource):
    """Yields the entire content of a single text file as one sample."""

    def __init__(self, file_path: Path):
        if not file_path.is_file():
            raise FileNotFoundError(f"Source file not found at: {file_path}")
        self._file_path = file_path

    def __len__(self) -> int:
        """Returns the number of samples (always 1 for this class)."""
        return 1

    def __iter__(self) -> Iterator[str]:
        with open(self._file_path, encoding="utf-8", errors="ignore") as f:
            yield f.read()


class FolderDataSource(DataSource):
    """Iterates through a directory and yields the content of each file."""

    def __init__(self, folder_path: Path):
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {folder_path}")

        self._file_paths = [p for p in folder_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
        print(f"✅ Found {len(self._file_paths)} files to process in {folder_path}.")

    def __len__(self) -> int:
        """Returns the total number of files found."""
        return len(self._file_paths)

    def __iter__(self) -> Iterator[str]:
        for file_path in tqdm(self._file_paths, desc="Reading source files"):
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                yield f.read()


class LineByLineFileDataSource(DataSource):
    """Reads a text file and yields each line as a separate sample."""

    def __init__(self, file_path: Path):
        if not file_path.is_file():
            raise FileNotFoundError(f"Source file not found at: {file_path}")
        self._file_path = file_path

        print("Pre-counting lines for progress bar...")
        with open(self._file_path, encoding="utf-8", errors="ignore") as f:
            self._line_count = sum(1 for _ in f)

    def __len__(self) -> int:
        """Returns the total number of lines in the file."""
        return self._line_count

    def __iter__(self) -> Iterator[str]:
        with open(self._file_path, encoding="utf-8", errors="ignore") as f:
            yield from f
