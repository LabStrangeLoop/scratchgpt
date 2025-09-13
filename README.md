# ScratchGPT

![ScratchGPT](https://raw.githubusercontent.com/LabStrangeLoop/scratchgpt/main/assets/logo.webp)

<p align="center">
  <a href="https://pypi.org/project/scratchgpt/"><img src="https://img.shields.io/pypi/v/scratchgpt.svg" alt="PyPI version"></a>
  <a href="https://github.com/LabStrangeLoop/scratchgpt/actions/workflows/tests.yml"><img src="https://github.com/LabStrangeLoop/scratchgpt/actions/workflows/tests.yml/badge.svg" alt="Tests Status"></a>
  <a href="https://github.com/LabStrangeLoop/scratchgpt/actions/workflows/lint.yml"><img src="https://github.com/LabStrangeLoop/scratchgpt/actions/workflows/lint.yml/badge.svg" alt="Lint Status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/scratchgpt/"><img src="https://img.shields.io/pypi/pyversions/3.12.svg" alt="Python versions"></a>
</p>

ScratchGPT is a Python project that implements a small-scale transformer-based
language model from scratch. It is designed for educational purposes, allowing
developers to explore the internals of a transformer model without the
complexity of large-scale frameworks. The project provides functionality for
training the model on custom datasets and generating text from a prompt.


## Why?

We want to allow people to experiment easily with any sequence-to-sequence
problems. This package is simple to understand, simple to use - show us your
projects using ScratchGPT.


## Features

- Custom transformer architecture implementation
- Training on user-provided text data
- Text generation using the trained model
- Command-line interfaces for training and inference

## Key Features

- **Custom Transformer Architecture**: A from-the-ground-up implementation of a decoder-only transformer, including Multi-Head Self-Attention , Feed-Forward layers, and Layer Normalization.
- **Flexible Tokenization**: Includes a simple character-level tokenizer and a wrapper for using any tokenizer from the Hugging Face Hub.
- **Configurable Training**: Easily configure model architecture (e.g., embedding_size, num_heads) and training parameters (e.g., learning_rate, batch_size) via a scratch_gpt.yaml file.
- **Command-Line Interfaces**: Comes with user-friendly CLIs for both training the model and performing inference.
- **Pre-tokenization Caching**: Caches tokenized datasets to disk for significantly faster startup on subsequent training runs.


## Requirements

- Python 3.12+
- `uv` for dependency management

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/LabStrangeLoop/scratchgpt.git
   cd scratchgpt
   ```

2. Install dependencies using uv:
   ```
   uv sync --all-groups
   ```

3. Install from pip:
   ```
   pip install scratchgpt
   ```


## Full Usage Examples

Please take a look at the ![simple example](./examples/simple.py) in the examples folder.

## Usage

### Training

To train the model on your custom dataset, run the `train` command. This will create an experiment folder containing the model weights, tokenizer files, and configuration.

```
uv run train -t <path_to_training_data> -e <experiment_folder>
```

- `-d, --data_source`: Path to the training data file or folder
- `-e, --experiment`: Path to the folder where experiment checkpoints will be saved
- `-t, --tokenizer`: (Optional) The Hugging Face Hub tokenizer to use (default: "gpt2")

### Inference

To generate text using a trained model, use `infer` command:

```
uv run infer -e <experiment_folder> [-dv <device>] [-m <max_tokens>]
```

- `-e, --experiment`: Path to the folder containing the trained model
- `-dv, --device`: Device to run the model on (default: "cuda")
- `-m, --max_tokens`: Maximum number of tokens to generate (default: 512)

### Tokenization

This project allows you to create your own tokenizers easily or bootstraps huggingface tokenizers for you to use.

## Project Structure

The repository is organized to separate concerns, making it easy to navigate.

- `scratchgpt/train.py`: Main training script.
- `scratchgpt/infer.py`: Inference script for text generation.
- `scratchgpt/config.py`: Contains all Pydantic configuration models.
- `scratchgpt/model/model.py`: The core Transformer model implementation.
- `scratchgpt/training/trainer.py`: Orchestrates the training and validation loops.
- `scratchgpt/tokenizer/`: Tokenizer implementations, including wrappers for Hugging Face.
- `scratchgpt/model_io.py`: Utilities for saving and loading models and tokenizers.
- `tests/`: Unit tests for the project.


## Development

This project uses various development tools:

- `mypy` for static type checking
- `ruff` for formatting and standard adherence
- `pytest` for testing

Run the following commands to ensure code quality:

```
uv run ruff --fix .
uv run mypy scratchgpt
uv run pytest ./tests/
```


## Future Roadmap

- [ ] Apply SOTA optimizations


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Authors

- Aleksandr Yeganov
- Dario Cazzani
