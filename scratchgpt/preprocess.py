import io
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import DTypeLike
from tqdm import tqdm

from .tokenizer.base_tokenizer import Tokenizer


class SupportsUpdate(Protocol):
    def update(self, n: int) -> Any: ...


class Preprocessor(Protocol):
    """
    Preprocessor protocol for handling dataset conversion using a specific tokenizer.
    """

    def __call__(
        self,
        source: io.TextIOBase,
        sink: io.BufferedIOBase,
        chunk_size: int,
        pbar: SupportsUpdate | None = None,
    ) -> None:
        """
        Process the input text source and write the result to the binary sink.
        Optionally updates a tqdm progress bar.
        """


class FilePreprocessor(Protocol):
    """
    Preprocessor that deals specifically with file system io.
    """

    def __call__(self, input_path: Path, output_path: Path, chunk_size: int = 10 * 1024 * 1024) -> None:
        """
        Process input and output paths
        """


class TokenizerPreprocessor(Preprocessor):
    """
    Default pre-processor. Tokenizes a text stream and writes the output
    to a binary stream, managing progress updates internally.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        vocab_size = self.tokenizer.vocab_size
        if vocab_size < 2**8:
            self.dtype: DTypeLike = np.uint8
        elif vocab_size < 2**16:
            self.dtype = np.uint16
        elif vocab_size < 2**32:
            self.dtype = np.uint32
        else:
            self.dtype = np.uint64
        print(f"Preprocessor initialized. Selected {np.dtype(self.dtype).name} for token storage.")

    def __call__(
        self,
        source: io.TextIOBase,
        sink: io.BufferedIOBase,
        chunk_size: int = 10 * 1024 * 1024,
        pbar: SupportsUpdate | None = None,
    ) -> None:
        """
        Reads from the source stream, tokenizes content in chunks, writes to the
        sink stream, and updates the provided progress bar.
        """
        while chunk := source.read(chunk_size):
            tokens = self.tokenizer.encode(chunk)
            token_array = np.array(tokens, dtype=self.dtype)
            sink.write(token_array.tobytes())
            if pbar:
                pbar.update(len(chunk.encode("utf-8", errors="ignore")))


class File2FileTokenizerPreprocessor(FilePreprocessor):
    """
    Orchestrates preprocessing for a single source file to a single destination file.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._preprocessor = TokenizerPreprocessor(tokenizer)

    def __call__(self, input_path: Path, output_path: Path, chunk_size: int = 10 * 1024 * 1024) -> None:
        if not input_path.is_file():
            raise ValueError(f"Input path must be a file: {input_path}")
        if output_path.exists():
            raise FileExistsError(f"Output path already exists: {output_path}")

        total_size = input_path.stat().st_size

        with (
            open(input_path, encoding="utf-8", errors="ignore") as source,
            open(output_path, "wb") as sink,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Tokenizing {input_path.name}") as pbar,
        ):
            self._preprocessor(source, sink, chunk_size, pbar)

        print(f"Successfully preprocessed '{input_path}' to '{output_path}'")


class Folder2FileTokenizerPreprocessor(FilePreprocessor):
    """
    Orchestrates preprocessing for a directory of source files to a single destination file.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._preprocessor = TokenizerPreprocessor(tokenizer)

    def __call__(self, input_path: Path, output_path: Path, chunk_size: int = 10 * 1024 * 1024) -> None:
        if not input_path.is_dir():
            raise ValueError(f"Input path must be a directory: {input_path}")
        if output_path.exists():
            raise FileExistsError(f"Output path already exists: {output_path}")

        files_to_process = [p for p in input_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
        total_size = sum(p.stat().st_size for p in files_to_process)

        print(f"Found {len(files_to_process)} files to process.")

        with (
            open(output_path, "wb") as sink,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Tokenizing Folder '{input_path.name}'") as pbar,
        ):
            for file_path in files_to_process:
                pbar.set_postfix_str(f"Processing: {file_path.name}", refresh=True)
                with open(file_path, encoding="utf-8", errors="ignore") as source:
                    self._preprocessor(source, sink, chunk_size, pbar)

        print(f"\nSuccessfully preprocessed folder '{input_path}' to '{output_path}'")
