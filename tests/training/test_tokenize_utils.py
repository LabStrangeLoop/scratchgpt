import pytest
from datasets import Dataset

from scratchgpt.tokenizer.char_tokenizer import CharTokenizer
from scratchgpt.training.tokenize_utils import (
    create_tokenize_function,
    prepare_dataset_for_training,
)


@pytest.fixture
def simple_tokenizer() -> CharTokenizer:
    """A simple character tokenizer. Includes a space, so 'a' is token 1."""
    return CharTokenizer(text="abcdefghijklmnopqrstuvwxyz ")


def test_tokenize_and_chunk_logic(simple_tokenizer):
    """Tests the core logic of the tokenize_and_chunk inner function."""
    block_size = 8
    text_column = "text"
    # This text has 26 chars. With block_size=8, chunk_size=9. 26 // 9 = 2.
    # We should get 2 full blocks.
    sample_data = {"text": ["abcdefghijklmnopqrstuvwxyz"]}

    tokenize_fn = create_tokenize_function(simple_tokenizer, block_size, text_column)
    processed = tokenize_fn(sample_data)

    assert "input_ids" in processed
    assert "labels" in processed

    assert len(processed["input_ids"]) == 2
    assert len(processed["labels"]) == 2

    # Check block content. With a space in the vocab, 'a' is 1, 'b' is 2, etc.
    assert processed["input_ids"][0] == list(range(1, 9))  # a-h
    assert processed["labels"][0] == list(range(2, 10))  # b-i

    # Second block starts after the first 9-token chunk (a-i).
    # It begins with 'j', which is token 10.
    assert processed["input_ids"][1] == list(range(10, 18))  # j-q
    assert processed["labels"][1] == list(range(11, 19))  # k-r


def test_tokenize_and_chunk_edge_cases(simple_tokenizer):
    """Tests scenarios with insufficient text."""
    block_size = 10
    text_column = "text"
    tokenize_fn = create_tokenize_function(simple_tokenizer, block_size, text_column)

    # Case 1: Text is too short to form any blocks
    short_text = {"text": ["abcdefghij"]}  # 10 chars, needs 11 for one block
    processed_short = tokenize_fn(short_text)
    assert len(processed_short["input_ids"]) == 0
    assert len(processed_short["labels"]) == 0

    # Case 2: Text is just long enough for one block
    exact_text = {"text": ["abcdefghijk"]}  # 11 chars
    processed_exact = tokenize_fn(exact_text)
    assert len(processed_exact["input_ids"]) == 1
    assert processed_exact["input_ids"][0] == list(range(1, 11))  # a-j


def test_prepare_dataset_for_training(simple_tokenizer):
    """Ensures the full preparation pipeline works as expected."""
    block_size = 5
    text_column = "content"
    # Concatenated text is 18 chars long. 18 // (5+1) = 3 blocks
    dataset = Dataset.from_dict({"content": ["abcdefghijkl", "mnopqr"]})

    processed_dataset = prepare_dataset_for_training(dataset, simple_tokenizer, block_size, text_column, num_proc=1)

    assert processed_dataset.column_names == ["input_ids", "labels"]
    assert processed_dataset.format["type"] == "torch"
    assert len(processed_dataset) == 3
