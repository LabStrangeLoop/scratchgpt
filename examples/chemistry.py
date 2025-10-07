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

Usage:
    python chemistry.py
"""

import sys
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
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


def load_reaction_dataset():
    """
    Load the USPTO-50k chemical reaction dataset from HuggingFace.

    This dataset contains ~50,000 reactions extracted from US patents,
    represented in SMILES notation with atom mapping.
    """
    print("Loading USPTO-50k reaction dataset from HuggingFace...")
    print("This dataset contains 50,000 chemical reactions from US patents.")

    # Load the dataset - it has train/val/test splits
    dataset = load_dataset("pingzhili/uspto-50k", split="train")

    print(f"✓ Loaded {len(dataset):,} reactions")
    return dataset


def prepare_reaction_text(dataset):
    """
    Extract reaction SMILES and concatenate them into training text.

    Reactions are in the format: reactants >> products
    Example: CC(=O)O.CCO >> CC(=O)OCC.O (esterification)

    We wrap each reaction with special tokens:
    [BOS] reaction [EOS]
    """
    print("\nPreparing reaction data for training...")

    # Determine which column contains the reactions
    column_names = dataset.column_names
    print(f"Dataset columns: {column_names}")

    # Try different possible column names (in priority order)
    reaction_column = None
    possible_columns = ['rxn_smiles', 'reaction_smiles', 'text', 'reaction', 'smiles', 'rxn']

    for col in possible_columns:
        if col in column_names:
            reaction_column = col
            break

    if reaction_column is None:
        # If no standard column found, print available columns and exit
        print(f"ERROR: Could not find reaction column!")
        print(f"Available columns: {column_names}")
        print("Please check the dataset structure.")
        return ""

    print(f"Using column: '{reaction_column}'")

    # Extract reactions and wrap with special tokens
    reactions = []
    for example in dataset:
        # Each example has the reaction in SMILES format
        reaction = str(example[reaction_column]).strip()
        if reaction and '>>' in reaction:  # Validate it's a proper reaction
            # Wrap with special tokens
            wrapped_reaction = f"[BOS]{reaction}[EOS]"
            reactions.append(wrapped_reaction)

    print(f"Extracted {len(reactions):,} valid reactions")

    # Show some example reactions
    print("\nSample reactions (with special tokens):")
    for i, reaction in enumerate(reactions[:3], 1):
        # Truncate long reactions for display
        display = reaction[:80] + "..." if len(reaction) > 80 else reaction
        print(f"  {i}. {display}")

    # Join all reactions with newlines
    full_text = "\n".join(reactions)
    print(f"\nTotal text length: {len(full_text):,} characters")

    return full_text


def create_chemistry_config(vocab_size):
    """
    Create a configuration optimized for chemical reaction prediction.

    Chemistry has different patterns than language or chess:
    - Reactions can be long (100-300 characters)
    - Pattern complexity is between chess and natural language
    - Needs to learn molecular substructure relationships
    """
    architecture = ScratchGPTArchitecture(
        block_size=256,      # Handle reactions up to ~256 characters
        embedding_size=256,  # Balanced for chemical pattern complexity
        num_heads=8,         # Good multi-head attention for chemistry
        num_blocks=6,        # Sufficient depth for reaction mechanisms
        vocab_size=vocab_size,
    )

    training = ScratchGPTTraining(
        max_epochs=15,       # Chemical patterns learn relatively quickly
        learning_rate=3e-4,  # Standard learning rate
        batch_size=32,       # Good batch size for reactions
        dropout_rate=0.1,    # Lower dropout for structured chemistry
        random_seed=1337,
        iteration_type="chunking",
    )

    return ScratchGPTConfig(architecture=architecture, training=training)


def generate_reaction_products(device, model, tokenizer, reactants, max_tokens=150):
    """
    Generate reaction products from given reactants.

    The model completes the reaction by predicting what comes after '>>'.
    We start with [BOS] and stop when we hit [EOS].
    """
    model.eval()

    # Get the [EOS] token ID for stopping generation
    eos_token_id = tokenizer.encode("[EOS]")[0] if "[EOS]" in tokenizer.vocabulary else None

    # Start with [BOS] and add the reaction arrow
    prompt = "[BOS]" + reactants + " >> "

    with torch.no_grad():
        context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=0.8,
            stop_token=eos_token_id
        )
        result = tokenizer.decode(generated[0].tolist())

    return result


def main():
    print("Chemical Reaction Prediction with ScratchGPT")
    print("=" * 60)

    # Step 1: Load dataset from HuggingFace
    print("\n--- Loading Chemical Reaction Dataset ---")
    dataset = load_reaction_dataset()

    # Step 2: Prepare reaction text
    reactions_text = prepare_reaction_text(dataset)

    if not reactions_text.strip():
        print("ERROR: No valid reactions were extracted!")
        sys.exit(1)

    # Step 3: Create character-level tokenizer
    print("\n--- Creating Character Tokenizer ---")
    tokenizer = CharTokenizer(text=reactions_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Includes special tokens: [BOS] (begin) and [EOS] (end)")
    print(f"Sample characters: {sorted(tokenizer.vocabulary)[:20]}")

    # Step 4: Create chemistry-optimized config
    print("\n--- Creating Chemistry Model Configuration ---")
    config = create_chemistry_config(tokenizer.vocab_size)
    print(
        f"Model: {config.architecture.embedding_size}D embeddings, "
        f"{config.architecture.num_blocks} blocks, {config.architecture.num_heads} heads"
    )

    # Step 5: Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if device.type == "cpu":
        print("⚠️  WARNING: Training on CPU will be slow!")
        print("   Expected time: 1-2 hours per epoch")
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)

    model = TransformerLanguageModel(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Step 6: Setup training with temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        reactions_file = temp_path / "reactions.txt"
        experiment_dir = temp_path / "chemistry_experiment"

        # Save reactions to file for data source
        print("\nSaving reactions to temporary file...")
        with open(reactions_file, "w", encoding="utf-8") as f:
            f.write(reactions_text)

        # Create data source
        data_source = create_data_source(str(reactions_file))

        # Setup optimizer and trainer
        optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
        trainer = Trainer(
            model=model,
            config=config.training,
            optimizer=optimizer,
            experiment_path=experiment_dir,
            device=device,
        )

        # Save tokenizer
        save_tokenizer(experiment_dir, tokenizer)

        # Step 7: Training
        print("\n--- Starting Chemical Reaction Training ---")
        print("The model will learn to predict reaction products from reactants")
        print("Press Ctrl-C to stop training early and see predictions")

        start_time = time.time()

        try:
            trainer.train(data_source=data_source, tokenizer=tokenizer)
            print(f"\n✅ Training completed in {time.time() - start_time:.1f} seconds")
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted after {time.time() - start_time:.1f} seconds")
            print("Proceeding with reaction prediction demo...")

        # Step 8: Reaction Prediction Demo
        print("\n--- Chemical Reaction Prediction Demo ---")
        print("Testing the model's ability to predict reaction products")
        print("=" * 70)

        # Test reactions - simple, well-known reactions
        test_reactions = [
            "CC(=O)O.CCO",           # Acetic acid + ethanol (esterification)
            "c1ccccc1.Cl2",          # Benzene + chlorine (chlorination)
            "CC=C.HBr",              # Propene + HBr (addition)
            "CC(=O)Cl.N",            # Acetyl chloride + ammonia
            "CCO.[O]",               # Ethanol + oxygen (oxidation)
        ]

        reaction_names = [
            "Esterification (acetic acid + ethanol)",
            "Chlorination (benzene + chlorine)",
            "Addition (propene + HBr)",
            "Amide formation (acetyl chloride + ammonia)",
            "Oxidation (ethanol + oxygen)",
        ]

        for reactants, name in zip(test_reactions, reaction_names):
            print(f"\nReaction: {name}")
            print(f"Reactants: {reactants}")
            print("-" * 50)

            result = generate_reaction_products(device, model, tokenizer, reactants)

            # Clean up the result by removing special tokens
            result = result.replace("[BOS]", "").replace("[EOS]", "")

            # Extract the predicted products (everything after >>)
            if ">>" in result:
                predicted_products = result.split(">>", 1)[1].strip()
                # Show first 80 characters of prediction
                display = predicted_products[:80] + "..." if len(predicted_products) > 80 else predicted_products
                print(f"Predicted products: {display}")
            else:
                print("Generated: (incomplete prediction)")

        print("\n" + "=" * 70)
        print("Chemical reaction prediction training complete!")
        print("\nWhat the model learned:")
        print("- Patterns of how molecular structures transform in reactions")
        print("- Common functional group conversions (esters, amides, etc.)")
        print("- Product structures that typically result from given reactants")
        print("- When to stop generating (using [EOS] token)")
        print("- The model doesn't know chemistry rules, just statistical patterns!")

        print(f"\nExperiment saved temporarily to: {experiment_dir}")
        print("All files will be cleaned up when the script exits.")


if __name__ == "__main__":
    main()