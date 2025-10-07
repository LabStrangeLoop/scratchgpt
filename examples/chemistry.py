#!/usr/bin/env python3
"""
Chemical Reaction Prediction Example - Train a transformer to predict reaction products

This script demonstrates training a GPT-style model on chemical reactions from the USPTO
patent database. It downloads reactions in SMILES notation and trains a transformer to
complete reactions by predicting products from reactants.

The model learns chemical transformation patterns without knowing any chemistry - it just
sees that certain molecular structures tend to produce other structures.

We use special tokens [BOS] and [EOS] to mark the beginning and end of each reaction,
helping the model learn when to stop generating.

"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from torch.optim import AdamW

from scratchgpt import (
    CharTokenizer,
    ScratchGPTArchitecture,
    ScratchGPTConfig,
    ScratchGPTTraining,
    Trainer,
    TransformerLanguageModel,
    save_tokenizer,
)
from scratchgpt.data import create_data_source

# Test reactions for demonstration
TEST_REACTIONS = [
    ("CC(=O)O.CCO", "Esterification (acetic acid + ethanol)"),
    ("c1ccccc1.Cl2", "Chlorination (benzene + chlorine)"),
    ("CC=C.HBr", "Addition (propene + HBr)"),
    ("CC(=O)Cl.N", "Amide formation (acetyl chloride + ammonia)"),
    ("CCO.[O]", "Oxidation (ethanol + oxygen)"),
]

# Display configuration
MAX_DISPLAY_LENGTH: int = 80
SEPARATOR_WIDTH: int = 70

# Summary text
TRAINING_SUMMARY = """
Chemical reaction prediction training complete!

What the model learned:
- Patterns of how molecular structures transform in reactions
- Common functional group conversions (esters, amides, etc.)
- Product structures that typically result from given reactants
- When to stop generating (using [EOS] token)
- The model doesn't know chemistry rules, just statistical patterns!
"""


def truncate_for_display(text: str, max_length: int = MAX_DISPLAY_LENGTH) -> str:
    """Truncate text for display, adding ellipsis if needed."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def load_reaction_dataset() -> Dataset:
    """
    Load the USPTO-50k chemical reaction dataset from HuggingFace.

    This dataset contains ~50,000 reactions extracted from US patents,
    represented in SMILES notation with atom mapping.
    """
    print("Loading USPTO-50k reaction dataset from HuggingFace...")
    print("This dataset contains 50,000 chemical reactions from US patents.")

    dataset: Dataset = load_dataset("pingzhili/uspto-50k", split="train")

    print(f"✓ Loaded {len(dataset):,} reactions")
    return dataset


def prepare_reaction_text(dataset: Dataset) -> str:
    """
    Extract reaction SMILES and concatenate them into training text.

    Reactions are in the format: reactants >> products
    Example: CC(=O)O.CCO >> CC(=O)OCC.O (esterification)

    We wrap each reaction with special tokens:
    [BOS] reaction [EOS]
    """
    print("\nPreparing reaction data for training...")

    column_names: list[str] = dataset.column_names
    print(f"Dataset columns: {column_names}")

    # Find reaction column
    possible_columns: list[str] = ["rxn_smiles", "reaction_smiles", "text", "reaction", "smiles", "rxn"]
    reaction_column: str | None = None

    for column in possible_columns:
        if column in column_names:
            reaction_column = column
            break

    if reaction_column is None:
        print("ERROR: Could not find reaction column!")
        print(f"Available columns: {column_names}")
        return ""

    print(f"Using column: '{reaction_column}'")

    # Extract and wrap reactions with special tokens
    reactions: list[str] = []
    for example in dataset:
        reaction: str = str(example[reaction_column]).strip()
        if reaction and ">>" in reaction:
            wrapped_reaction: str = f"[BOS]{reaction}[EOS]"
            reactions.append(wrapped_reaction)

    print(f"Extracted {len(reactions):,} valid reactions")

    # Show sample reactions
    print("\nSample reactions (with special tokens):")
    for index, reaction in enumerate(reactions[:3], start=1):
        display: str = truncate_for_display(reaction)
        print(f"  {index}. {display}")

    full_text: str = "\n".join(reactions)
    print(f"\nTotal text length: {len(full_text):,} characters")

    return full_text


def create_chemistry_config(vocab_size: int) -> ScratchGPTConfig:
    """
    Create a configuration optimized for chemical reaction prediction.

    Chemistry has different patterns than language or chess:
    - Reactions can be long (100-300 characters)
    - Pattern complexity is between chess and natural language
    - Needs to learn molecular substructure relationships
    """
    architecture: ScratchGPTArchitecture = ScratchGPTArchitecture(
        block_size=256,
        embedding_size=256,
        num_heads=8,
        num_blocks=6,
        vocab_size=vocab_size,
    )

    training: ScratchGPTTraining = ScratchGPTTraining(
        max_epochs=15,
        learning_rate=3e-4,
        batch_size=32,
        dropout_rate=0.1,
        random_seed=1337,
        iteration_type="chunking",
    )

    return ScratchGPTConfig(architecture=architecture, training=training)


def generate_reaction_products(
    device: torch.device,
    model: TransformerLanguageModel,
    tokenizer: CharTokenizer,
    reactants: str,
    max_tokens: int = 150,
) -> str:
    """
    Generate reaction products from given reactants.

    The model completes the reaction by predicting what comes after '>>'.
    We start with [BOS] and stop when we hit [EOS].
    """
    model.eval()

    # Clean reactants - SMILES shouldn't have spaces
    reactants_clean: str = reactants.strip().replace(" ", "")

    # Get the [EOS] token ID for stopping generation
    eos_in_vocab: bool = "[EOS]" in tokenizer.vocabulary
    eos_token_id: int | None = tokenizer.encode("[EOS]")[0] if eos_in_vocab else None

    # Build prompt with special tokens and reaction arrow
    prompt: str = f"[BOS]{reactants_clean}>>"

    with torch.no_grad():
        context: torch.Tensor = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        generated: torch.Tensor = model.generate(
            context, max_new_tokens=max_tokens, temperature=0.8, stop_token=eos_token_id
        )
        result: str = tokenizer.decode(generated[0].tolist())

    return result


def main() -> None:
    print("Chemical Reaction Prediction with ScratchGPT")
    print("=" * 60)

    # Load dataset from HuggingFace
    print("\n--- Loading Chemical Reaction Dataset ---")
    dataset: Dataset = load_reaction_dataset()

    # Prepare reaction text
    reactions_text: str = prepare_reaction_text(dataset)

    if not reactions_text.strip():
        print("ERROR: No valid reactions were extracted!")
        sys.exit(1)

    # Create character-level tokenizer
    print("\n--- Creating Character Tokenizer ---")
    tokenizer: CharTokenizer = CharTokenizer(text=reactions_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Includes special tokens: [BOS] (begin) and [EOS] (end)")
    sample_chars: list[str] = sorted(tokenizer.vocabulary)[:20]
    print(f"Sample characters: {sample_chars}")

    # Create chemistry-optimized config
    print("\n--- Creating Chemistry Model Configuration ---")
    config: ScratchGPTConfig = create_chemistry_config(tokenizer.vocab_size)
    print(
        f"Model: {config.architecture.embedding_size}D embeddings, "
        f"{config.architecture.num_blocks} blocks, {config.architecture.num_heads} heads"
    )

    # Setup device and model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model: TransformerLanguageModel = TransformerLanguageModel(config)
    model = model.to(device)
    total_params: int = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Setup training with temporary directory
    temp_path: Path = Path(tempfile.mkdtemp())
    reactions_file: Path = temp_path / "reactions.txt"
    experiment_dir: Path = temp_path / "chemistry_experiment"

    print("\nSaving reactions to temporary file...")
    with open(reactions_file, "w", encoding="utf-8") as file:
        file.write(reactions_text)

    data_source = create_data_source(str(reactions_file))
    optimizer: AdamW = AdamW(model.parameters(), lr=config.training.learning_rate)
    trainer: Trainer = Trainer(
        model=model,
        config=config.training,
        optimizer=optimizer,
        experiment_path=experiment_dir,
        device=device,
    )

    save_tokenizer(experiment_dir, tokenizer)

    # Training
    print("\n--- Starting Chemical Reaction Training ---")
    print("The model will learn to predict reaction products from reactants")
    print("Press Ctrl-C to stop training early and see predictions")

    start_time: float = time.time()

    try:
        trainer.train(data_source=data_source, tokenizer=tokenizer)
        training_time: float = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
    except KeyboardInterrupt:
        training_time: float = time.time() - start_time
        print(f"\n⚠️ Training interrupted after {training_time:.1f} seconds")
        print("Proceeding with reaction prediction demo...")

    # Prediction demo
    print("\n--- Chemical Reaction Prediction Demo ---")
    print("Testing the model's ability to predict reaction products")
    print("=" * SEPARATOR_WIDTH)

    for reactants, reaction_name in TEST_REACTIONS:
        print(f"\nReaction: {reaction_name}")
        print(f"Reactants: {reactants}")
        print("-" * 50)

        result: str = generate_reaction_products(device, model, tokenizer, reactants)

        # Clean up special tokens from result
        result_clean: str = result.replace("[BOS]", "").replace("[EOS]", "")

        # Extract predicted products (everything after >>)
        if ">>" in result_clean:
            predicted_products: str = result_clean.split(">>", 1)[1].strip()
            display: str = truncate_for_display(predicted_products)
            print(f"Predicted products: {display}")
        else:
            print("Generated: (incomplete prediction)")

    print("\n" + "=" * SEPARATOR_WIDTH)

    # Summary
    print(TRAINING_SUMMARY)
    print(f"Experiment saved temporarily to: {experiment_dir}")
    print("All files will be cleaned up when the script exits.")

    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    main()
