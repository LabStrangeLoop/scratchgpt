#!/usr/bin/env python3
"""
Chess Engine Training Example - Train a transformer to predict chess moves using ScratchGPT

This script demonstrates training a GPT-style model on chess games from the Lichess database.
It downloads a collection of games in PGN format, parses them into move sequences,
and trains a transformer to continue games by predicting the next moves.

The model learns chess patterns without knowing the rules - it just sees that certain
move sequences tend to follow others in master games from Lichess.

Usage:
    python chess.py
    python chess.py -g https://database.lichess.org/blitz/lichess_db_blitz_rated_2024-01.pgn.zst
"""

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
import zstandard as zstd
from torch.nn import functional as F
from torch.optim import AdamW

from examples.chess_tokenizer import ChessTokenizer
from scratchgpt import (
    ScratchGPTArchitecture,
    ScratchGPTConfig,
    ScratchGPTTraining,
    Trainer,
    TransformerLanguageModel,
    save_tokenizer,
)
from scratchgpt.data import create_data_source

# Alternative: use character-level tokenization
# from scratchgpt import CharTokenizer

# Default Lichess database file
DEFAULT_LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-02.pgn.zst"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a chess move predictor using ScratchGPT")
    parser.add_argument(
        "-g",
        "--game-url",
        type=str,
        default=DEFAULT_LICHESS_URL,
        help=f"Lichess database URL to download (default: {DEFAULT_LICHESS_URL})",
    )
    return parser.parse_args()


class ChessDataLoader:
    """Handles downloading and parsing of Lichess chess databases."""

    def __init__(self, game_url: str):
        self.game_url = game_url

    def download_and_parse(self) -> str:
        """Download, decompress, and parse chess games into clean move sequences."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            print(f"Working in temporary directory: {temp_path}")
            pgn_file = self._download_and_decompress(temp_path)
            games_text = self._parse_pgn_to_games(pgn_file)
            return games_text

    def _download_and_decompress(self, temp_dir: Path) -> Path:
        """Download and decompress the Lichess database file."""
        filename = Path(urlparse(self.game_url).path).name
        compressed_file = temp_dir / filename

        print(f"Downloading: {filename}")
        print("This may take several minutes depending on file size...")
        urlretrieve(self.game_url, compressed_file)

        pgn_file = temp_dir / filename.replace(".zst", "")
        print(f"Decompressing: {filename}")

        dctx = zstd.ZstdDecompressor()
        with open(compressed_file, "rb") as compressed_fp, open(pgn_file, "wb") as output_fp:
            dctx.copy_stream(compressed_fp, output_fp)

        # Remove compressed file to save space
        compressed_file.unlink()
        return pgn_file

    def _parse_pgn_to_games(self, pgn_file: Path) -> str:
        """Parse PGN file and extract move sequences."""
        print(f"Parsing games from: {pgn_file.name}")

        games = []
        current_game_lines = []
        games_processed = 0

        with open(pgn_file, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                if line_num % 100_000 == 0:
                    print(f"Processed {line_num:,} lines, found {games_processed:,} games")
                if line.startswith("["):
                    continue
                if not line:
                    continue

                current_game_lines.append(line)

                if any(result in line for result in ["1-0", "0-1", "1/2-1/2", "*"]):
                    game_text = " ".join(current_game_lines).strip()
                    clean_text = self._clean_game_text(game_text)
                    # Only keep games with more than 2 moves
                    if len(clean_text.split()) > 2:
                        games.append(clean_text)
                        games_processed += 1

                    # Reset for next game
                    current_game_lines = []

        print(f"Extracted {len(games)} valid games")
        return "\n".join(games)

    def _clean_game_text(self, game_text: str) -> str:
        """Clean annotations and comments from game text."""
        # Remove comments in curly braces
        game_text = re.sub(r"\{[^}]*\}", " ", game_text)

        # Remove evaluation annotations like [%eval 0.5]
        game_text = re.sub(r"\[%[^\]]*\]", " ", game_text)

        # Clean up multiple spaces
        game_text = re.sub(r"\s+", " ", game_text).strip()

        # Remove game results from the end
        for result in ["1-0", "0-1", "1/2-1/2", "*"]:
            if game_text.endswith(" " + result):
                game_text = game_text[: -len(" " + result)].strip()
                break

        return game_text


def create_chess_config(tokenizer_vocab_size: int) -> ScratchGPTConfig:
    """Create a configuration optimized for chess move prediction."""
    # Chess-optimized architecture
    architecture = ScratchGPTArchitecture(
        block_size=256,  # Longer context for chess games (can see ~60-80 moves)
        embedding_size=384,  # Balanced size for chess vocabulary
        num_heads=8,  # Good attention for chess patterns
        num_blocks=6,  # Sufficient depth for chess understanding
        vocab_size=tokenizer_vocab_size,
    )

    # Training config optimized for chess patterns
    training = ScratchGPTTraining(
        max_epochs=15,  # Chess patterns learn faster than language
        learning_rate=3e-4,  # Standard rate works well for chess
        batch_size=32,  # Good balance for chess sequences
        dropout_rate=0.1,  # Lower dropout for structured chess patterns
        random_seed=1337,
        iteration_type="chunking",
    )

    return ScratchGPTConfig(architecture=architecture, training=training)


def generate_chess_moves(
    device: torch.device,
    model: TransformerLanguageModel,
    tokenizer,
    game_start: str,
    max_moves: int = 8,
    temperature: float = 0.8,
) -> str:
    """
    Generate chess moves given the start of a game.

    Uses moderate temperature to balance chess-like patterns with some creativity.
    """
    model.eval()

    # Encode the game start
    context = torch.tensor(tokenizer.encode(game_start)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate tokens (approximately 4-6 tokens per move)
        for _ in range(max_moves * 6):
            # Crop context to model's block size
            cropped_context = context[:, -model._block_size :]

            # Get logits and apply temperature
            logits = model(cropped_context)
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to context
            context = torch.cat((context, next_token), dim=1)

            # Stop if we've generated enough content
            current_length = len(tokenizer.decode(context[0].tolist()))
            if current_length > len(game_start) + max_moves * 8:  # Rough estimate
                break

    return tokenizer.decode(context[0].tolist())


def main() -> None:
    print("Chess Move Prediction Training with ScratchGPT")
    print("=" * 60)

    # Parse arguments
    args = parse_args()

    # Step 1: Download and parse chess data
    print("\n--- Downloading and Parsing Chess Games ---")
    data_loader = ChessDataLoader(args.game_url)
    games_text = data_loader.download_and_parse()

    if not games_text.strip():
        print("ERROR: No games were parsed successfully!")
        sys.exit(1)

    # Show sample of parsed games
    sample_games = games_text.split("\n")[:3]
    print("\nSample parsed games:")
    for i, game in enumerate(sample_games, 1):
        preview = game[:80] + "..." if len(game) > 80 else game
        print(f"{i}: {preview}")

    # Step 2: Setup tokenizer
    print("\n--- Creating Chess Tokenizer ---")
    tokenizer = ChessTokenizer()
    print(f"Chess vocabulary size: {tokenizer.vocab_size:,}")

    # Alternative approach using character-level tokenization:
    # tokenizer = CharTokenizer(text=games_text)
    # print(f"Character vocabulary size: {tokenizer.vocab_size}")
    #
    # Trade-offs:
    # - ChessTokenizer: Domain-specific, understands chess moves as units (~10k vocab)
    # - CharTokenizer: General, treats chess as character sequences (~60 vocab)
    # - ChessTokenizer should learn chess patterns more efficiently

    # Step 3: Create chess-optimized configuration
    print("\n--- Creating Chess Model Configuration ---")
    config = create_chess_config(tokenizer.vocab_size)
    print(
        f"Model configuration: {config.architecture.embedding_size}D embeddings, "
        f"{config.architecture.num_blocks} blocks, {config.architecture.num_heads} heads"
    )
    # Step 4: Setup device and model
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

    # Step 5: Setup training
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    # Create temporary file for chess games and data source
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        chess_games_file = temp_path / "chess_games.txt"

        # Save parsed games to file
        with open(chess_games_file, "w", encoding="utf-8") as f:
            f.write(games_text)

        # Create data source using ScratchGPT's standard approach
        data_source = create_data_source(str(chess_games_file))

        # Create experiment directory
        experiment_dir = temp_path / "chess_experiment"

        # Create trainer
        trainer = Trainer(
            model=model, config=config.training, optimizer=optimizer, experiment_path=experiment_dir, device=device
        )

        # Save tokenizer
        save_tokenizer(experiment_dir, tokenizer)

        # Step 6: Training
        print("\n--- Starting Chess Training ---")
        print("The model will learn to predict chess moves based on grandmaster games")
        print("Press Ctrl-C to stop training early and proceed to move generation demo")

        start_time = time.time()

        try:
            trainer.train(data_source=data_source, tokenizer=tokenizer)
            print(f"\n✅ Training completed in {time.time() - start_time:.1f} seconds")
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted after {time.time() - start_time:.1f} seconds")
            print("Proceeding with chess move generation demo...")

        # Step 7: Chess Move Generation Demo
        print("\n--- Chess Move Generation Demo ---")
        model.eval()

        # Test with famous chess openings
        test_positions = [
            "1. e4 e5 2. Nf3",  # Italian Game start
            "1. d4 d5 2. c4",  # Queen's Gambit
            "1. e4 c5",  # Sicilian Defense
            "1. Nf3 Nf6 2. c4",  # English Opening
            "1. e4 e6 2. d4",  # French Defense
        ]

        print("Generating continuations for famous chess openings:")
        print("=" * 70)

        for position in test_positions:
            print(f"\nPosition: {position}")
            print("-" * 50)

            # Generate continuation
            continuation = generate_chess_moves(
                device=device, model=model, tokenizer=tokenizer, game_start=position + " ", max_moves=6, temperature=0.8
            )

            # Extract generated part
            generated_part = continuation[len(position) :].strip()

            # Show first several moves of continuation
            generated_moves = generated_part.split()[:12]  # Show ~6 moves
            if generated_moves:
                print(f"Continuation: {' '.join(generated_moves)}")
            else:
                print("Generated: (no valid continuation)")

        print("\n" + "=" * 70)
        print("Chess move prediction training complete!")
        print("\nWhat the model learned:")
        print("- Chess move patterns from thousands of grandmaster games")
        print("- Common responses to popular openings")
        print("- Typical piece development and tactical motifs")
        print("- The model doesn't know chess rules, just statistical patterns!")

        print(f"\nExperiment saved temporarily to: {experiment_dir}")
        print("All files will be cleaned up when the script exits.")


if __name__ == "__main__":
    main()
