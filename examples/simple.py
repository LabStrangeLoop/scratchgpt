#!/usr/bin/env python3
"""
Simple example showing minimal usage of ScratchGPT to train on Darwin's "On the Origin of Species"

This script demonstrates:
1. Downloading training data from Project Gutenberg
2. Setting up a basic configuration
3. Training a small transformer model
4. Basic text generation

Usage:
    python simple.py
"""

import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.optim import AdamW

# Import ScratchGPT components
from scratchgpt import (
    CharTokenizer,
    FileDataSource,
    ScratchGPTArchitecture,
    ScratchGPTConfig,
    ScratchGPTTraining,
    Trainer,
    TransformerLanguageModel,
)


def download_darwin_text(data_file: Path) -> None:
    """Download Darwin's 'On the Origin of Species' using Python's built-in urllib."""
    print("Downloading 'On the Origin of Species' by Charles Darwin...")
    url = "https://www.gutenberg.org/files/1228/1228-0.txt"

    try:
        urlretrieve(url, data_file)
        print(f"Downloaded data to: {data_file}")
    except Exception as e:
        print(f"Failed to download data: {e}")
        print("Please manually download the file from:")
        print(url)
        sys.exit(1)


def create_simple_config() -> ScratchGPTConfig:
    """Create a minimal configuration suitable for quick training."""
    # Small architecture for quick training on CPU/small GPU
    architecture = ScratchGPTArchitecture(
        block_size=128,
        embedding_size=256,
        num_heads=8,
        num_blocks=4,
        # vocab_size will be set based on the tokenizer
    )

    # Training config optimized for quick results
    training = ScratchGPTTraining(
        max_epochs=20,
        learning_rate=3e-4,
        batch_size=32,
        dropout_rate=0.1,
        random_seed=1337,
    )

    return ScratchGPTConfig(
        architecture=architecture,
        training=training
    )


def prepare_text_for_tokenizer(data_file: Path) -> str:
    """Read the text file for tokenization."""
    print(f"Reading text from: {data_file}")

    with open(data_file, encoding='utf-8') as f:
        text = f.read()

    print(f"Text length: {len(text):,} characters")
    return text


def main():
    print("ScratchGPT Simple Training Example")
    print("=" * 50)

    # Use temporary directory that auto-cleans when done
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_file = tmp_path / "darwin_origin_species.txt"
        experiment_dir = tmp_path / "darwin_experiment"

        # Step 1: Download data
        download_darwin_text(data_file)

        # Step 2: Prepare text and create tokenizer
        text = prepare_text_for_tokenizer(data_file)
        print("Creating character-level tokenizer...")
        tokenizer = CharTokenizer(text=text)
        print(f"Vocabulary size: {tokenizer.vocab_size}")

        # Alternative: Use a pre-trained tokenizer like GPT-2
        # This requires: pip install 'scratchgpt[hf-tokenizers]'
        #
        # from scratchgpt import HuggingFaceTokenizer
        # tokenizer = HuggingFaceTokenizer.from_hub("gpt2")
        # print(f"Vocabulary size: {tokenizer.vocab_size}")  # ~50,257 tokens
        #
        # Trade-offs:
        # - CharTokenizer: Small vocab (~100 chars), learns from scratch, simple
        # - GPT-2 Tokenizer: Large vocab (~50K tokens), pre-trained, better text quality
        # - GPT-2 tokenizer will likely generate more coherent text but requires more memory

        # Step 3: Create configuration
        config = create_simple_config()
        config.architecture.vocab_size = tokenizer.vocab_size
        print(f"Model configuration: {config.architecture.embedding_size}D embeddings, "
              f"{config.architecture.num_blocks} blocks, {config.architecture.num_heads} heads")

        # Step 4: Setup model and training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = TransformerLanguageModel(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
        data_source = FileDataSource(data_file)

        # Step 5: Create trainer and start training
        trainer = Trainer(
            model=model,
            config=config.training,
            optimizer=optimizer,
            experiment_path=experiment_dir,
            device=device
        )

        print("\nStarting training...")
        trainer.train(data=data_source, tokenizer=tokenizer)

        # Step 6: Simple text generation demo
        print("\nTesting text generation:")
        model.eval()

        test_prompts = [
            "Natural selection",
            "The origin of species",
            "Darwin observed"
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: '{prompt}'")
            context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=100)
                result = tokenizer.decode(generated[0].tolist())
                print(f"Generated: {result}")

        print("\nTraining complete! All temporary files automatically cleaned up.")
        print("Run the script again to start fresh.")


if __name__ == "__main__":
    main()
