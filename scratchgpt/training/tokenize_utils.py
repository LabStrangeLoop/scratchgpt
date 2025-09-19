from collections.abc import Callable
from typing import Any

from datasets import Dataset

from scratchgpt.tokenizer.base_tokenizer import Tokenizer

TokenizeFunc = Callable[[dict[str, list[Any]]], dict[str, list[list[int]]]]


def create_tokenize_function(
    tokenizer: Tokenizer,
    block_size: int,
    text_column: str,
) -> TokenizeFunc:
    """
    Creates a tokenization function for use with `datasets.map`.

    This function tokenizes all texts, concatenates them, and then splits
    the result into chunks of `block_size`.
    """

    def tokenize_and_chunk(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        """Tokenizes a batch of texts and forms fixed-size blocks."""
        # Ensure the specified text column exists.
        if text_column not in examples:
            raise KeyError(
                f"Text column '{text_column}' not found in dataset. Available columns: {list(examples.keys())}"
            )

        # 1. Concatenate all tokens from all documents in the batch.
        all_tokens: list[int] = []
        for text in examples[text_column]:
            all_tokens.extend(tokenizer.encode(text))

        # 2. Create chunks of `block_size + 1` to get both inputs and labels.
        chunk_size = block_size + 1
        total_tokens = len(all_tokens)

        # If we can't form at least one full chunk, there's nothing to do.
        if total_tokens < chunk_size:
            return {"input_ids": [], "labels": []}

        # 3. Drop the last partial chunk to ensure all sequences are full length.
        num_to_drop = total_tokens % chunk_size
        if num_to_drop > 0:
            all_tokens = all_tokens[:-num_to_drop]

        # 4. Create input_ids and labels from the chunks.
        input_ids = [all_tokens[i : i + block_size] for i in range(0, len(all_tokens), chunk_size)]
        labels = [all_tokens[i + 1 : i + chunk_size] for i in range(0, len(all_tokens), chunk_size)]

        return {"input_ids": input_ids, "labels": labels}

    return tokenize_and_chunk


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer: Tokenizer,
    block_size: int,
    text_column: str,
    num_proc: int | None = None,
) -> Dataset:
    """
    Prepares a dataset for training by tokenizing and chunking it.
    """
    tokenize_fn = create_tokenize_function(tokenizer=tokenizer, block_size=block_size, text_column=text_column)

    columns_to_remove = list(dataset.features)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
    )

    # Set the dataset format for PyTorch for direct use in DataLoaders.
    tokenized_dataset.set_format("torch", columns=["input_ids", "labels"])

    return tokenized_dataset
