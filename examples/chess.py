#!/usr/bin/env python3
"""
Chess Database Parser - Download and parse Lichess chess games

Downloads a Lichess PGN database, decompresses it, and extracts clean move sequences.
Uses temporary directory for all operations, just like simple.py.

Usage:
    python chess.py
    python chess.py -g https://database.lichess.org/blitz/lichess_db_blitz_rated_2024-01.pgn.zst
"""

import argparse
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import zstandard as zstd

DEFAULT_LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-02.pgn.zst"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and parse Lichess chess games database")
    parser.add_argument(
        "-g",
        "--game-url",
        type=str,
        default=DEFAULT_LICHESS_URL,
        help=f"Lichess database URL to download (default: {DEFAULT_LICHESS_URL})",
    )
    return parser.parse_args()


def clean_game_text(game_text: str) -> str:
    """Clean annotations and comments from game text."""
    import re

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


def download_and_decompress(url: str, temp_dir: Path) -> Path:
    """Download and decompress the Lichess database file."""
    # Download
    filename = Path(urlparse(url).path).name
    compressed_file = temp_dir / filename

    print(f"Downloading: {filename}")
    urlretrieve(url, compressed_file)

    pgn_file = temp_dir / filename.replace(".zst", "")
    print(f"Decompressing: {filename}")

    dctx = zstd.ZstdDecompressor()

    with open(compressed_file, "rb") as compressed_fp, open(pgn_file, "wb") as output_fp:
        dctx.copy_stream(compressed_fp, output_fp)

    # Remove compressed file
    compressed_file.unlink()
    return pgn_file


def parse_pgn_to_games(pgn_file: Path) -> str:
    """Parse PGN file and extract move sequences."""
    print(f"Parsing games from: {pgn_file.name}")

    games: list[str] = []
    current_game_lines: list[str] = []
    games_processed: int = 0

    with open(pgn_file, encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line_num % 100_000 == 0:
                print(f"Processed {line_num:,} lines, found {games_processed:,} games")
            if line.startswith("["):  # Skip metadata lines (lines starting with [)
                continue
            if not line:  # Skip empty lines
                continue

            current_game_lines.append(line)  # Add to current game

            # Check if game ended (contains result)
            if any(result in line for result in ["1-0", "0-1", "1/2-1/2", "*"]):
                # Join all moves for this game
                game_text = " ".join(current_game_lines).strip()
                clean_text = clean_game_text(game_text)

                if len(clean_text.split()) > 2:  # More than just the result and more than one move
                    games.append(clean_text)
                    games_processed += 1

                current_game_lines = []

    print(f"Extracted {len(games)} valid games")
    return "\n".join(games)


def main():
    args = parse_args()

    # Use temporary directory for all operations
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        print(f"Working in temporary directory: {temp_path}")

        pgn_file = download_and_decompress(args.game_url, temp_path)
        games_text = parse_pgn_to_games(pgn_file)

        processed_file = temp_path / "chess_games.txt"
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(games_text)

        sample_games = games_text.split("\n")[:3]
        print("\nSample games:")
        for i, game in enumerate(sample_games, 1):
            preview = game[:100] + "..." if len(game) > 100 else game
            print(f"{i}: {preview}")

        print(f"\nProcessed {len(games_text.split()):,} games")
        print(f"Data ready for training: {processed_file}")
        print("All temporary files will be cleaned up when script exits.")


if __name__ == "__main__":
    main()
