from typing import Any

from scratchgpt.data.datasource import DataSource
from scratchgpt.data.hf_datasource import HFDataSource


def create_data_source(
    path_or_name: str,
    split: str = "train",
    streaming: bool = False,
    text_column: str = "text",
    **kwargs: Any,
) -> DataSource:
    """
    Create a data source from a path or dataset name.

    Examples:
        # HuggingFace Hub dataset
        >>> ds = create_data_source("wikitext-2-raw-v1")

        # Local text file
        >>> ds = create_data_source("data.txt")

        # Local CSV file
        >>> ds = create_data_source("data.csv", text_column="content")

        # Folder of text files
        >>> ds = create_data_source("./texts/")

        # Streaming large dataset
        >>> ds = create_data_source("openwebtext", streaming=True)

    Args:
        path_or_name: HF Hub dataset name or path to local data
        split: Dataset split to use
        streaming: Whether to use streaming mode
        text_column: Column name containing text
        **kwargs: Additional arguments for HFDataSource

    Returns:
        DataSource instance
    """
    return HFDataSource(
        path_or_name=path_or_name,
        split=split,
        streaming=streaming,
        text_column=text_column,
        **kwargs,
    )


__all__ = ["DataSource", "HFDataSource", "create_data_source"]
