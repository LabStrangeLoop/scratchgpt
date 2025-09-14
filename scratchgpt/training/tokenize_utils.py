from collections.abc import Callable
from typing import Any

from datasets import Dataset

from scratchgpt.tokenizer.base_tokenizer import Tokenizer

TokenizeFunc = Callable[[dict[str, list[Any]]], dict[str, list[list[int]]]]


def create_tokenize_function(
    tokenizer: Tokenizer,
    block_size: int,
    text_column: str = "text",
) -> TokenizeFunc:
    """
    Create a tokenization function for HF datasets.map().

    Args:
        tokenizer: Tokenizer to use
        block_size: Size of text blocks for training
        text_column: Name of column containing text

    Returns:
        Function suitable for dataset.map()
    """

    def tokenize_and_chunk(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        """Tokenize text and chunk into blocks."""
        texts: Any = examples.get(text_column, examples)

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        # Chunk into blocks
        blocks = []
        for i in range(0, len(all_tokens) - block_size, block_size + 1):
            input_ids = all_tokens[i : i + block_size]
            labels = all_tokens[i + 1 : i + block_size + 1]

            if len(input_ids) == block_size:  # Only keep full blocks
                blocks.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )

        # Return in format expected by HF datasets
        if blocks:
            return {
                "input_ids": [b["input_ids"] for b in blocks],
                "labels": [b["labels"] for b in blocks],
            }
        else:
            return {"input_ids": [], "labels": []}

    return tokenize_and_chunk


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer: Tokenizer,
    block_size: int,
    num_proc: int | None = None,
) -> Dataset:
    """
    Prepare a dataset for training by tokenizing and chunking.

    Args:
        dataset: HF dataset to prepare
        tokenizer: Tokenizer to use
        block_size: Size of text blocks
        num_proc: Number of processes for parallel processing

    Returns:
        Processed dataset ready for training
    """
    # Determine text column
    text_column = "text"
    has_column_names = hasattr(dataset, "column_names") and dataset.column_names

    if has_column_names and "text" not in dataset.column_names:
        text_column = dataset.column_names[0]

    # Create tokenization function
    tokenize_fn = create_tokenize_function(tokenizer, block_size, text_column)

    # Remove original columns and keep only input_ids and labels
    columns_to_remove = dataset.column_names if hasattr(dataset, "column_names") else []

    # Tokenize and chunk dataset
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
    )

    # Set format for PyTorch
    tokenized_dataset.set_format("torch", columns=["input_ids", "labels"])

    return tokenized_dataset
