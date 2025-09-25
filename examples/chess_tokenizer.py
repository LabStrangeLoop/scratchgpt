import json
import re
from pathlib import Path
from typing import Self, override

from scratchgpt.tokenizer.base_tokenizer import SerializableTokenizer, register_tokenizer


@register_tokenizer("ChessTokenizer")
class ChessTokenizer(SerializableTokenizer):
    """
    A deterministic tokenizer for chess games in Standard Algebraic Notation (SAN).

    This tokenizer uses a pre-generated, fixed vocabulary that covers all
    syntactically valid SAN moves, move numbers, and special PGN symbols.
    It does not learn from data but is designed to be comprehensive for the
    domain of chess.
    """

    def __init__(self, vocab: list[str] | None = None) -> None:
        if vocab is not None:
            self._vocabulary = vocab
        else:
            self._vocabulary = self._create_vocabulary()

        self._encoding_mapping = {token: i for i, token in enumerate(self._vocabulary)}
        self._decoding_mapping = dict(enumerate(self._vocabulary))

    @staticmethod
    def _create_vocabulary() -> list[str]:
        """Generates the complete, deterministic vocabulary for chess."""
        # Control and special tokens
        tokens = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "1-0", "0-1", "1/2-1/2", "*", "+", "#", "O-O", "O-O-O"}

        # Move numbers (1. to 300. and 1... to 300...)
        for i in range(1, 301):
            tokens.add(f"{i}.")
            # not adding tokens for move fragments
        #            tokens.add(f"{i}...")

        pieces = ["N", "B", "R", "Q", "K"]
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
        squares = [f + r for f in files for r in ranks]
        promotions = ["=Q", "=R", "=B", "=N"]

        # Pawn moves and captures
        for square in squares:
            tokens.add(square)  # e.g., e4
            for file in files:
                tokens.add(file + "x" + square)  # e.g., dxe5

        # Pawn promotions and capture-promotions
        promo_ranks = {"1", "8"}
        for square in squares:
            if square[1] in promo_ranks:
                for p_piece in promotions:
                    tokens.add(square + p_piece)  # e.g., e8=Q
                    for file in files:
                        tokens.add(file + "x" + square + p_piece)  # e.g., dxe8=Q

        # Piece moves (including disambiguation)
        for piece in pieces:
            for square in squares:
                tokens.add(piece + square)
                tokens.add(piece + "x" + square)
                for file in files:
                    tokens.add(piece + file + square)
                    tokens.add(piece + file + "x" + square)
                for rank in ranks:
                    tokens.add(piece + rank + square)
                    tokens.add(piece + rank + "x" + square)

        return sorted(tokens)

    @override
    def encode(self, text: str) -> list[int]:
        # Add space after move numbers (e.g., "1.e4" -> "1. e4")
        processed_text = re.sub(r"(\d+\.{1,3})", r"\1 ", text)
        # Add spaces around check/mate symbols to ensure they are tokenized separately
        processed_text = processed_text.replace("+", " + ").replace("#", " # ")

        raw_tokens = [token for token in processed_text.split() if token]

        unk_token_id = self._encoding_mapping["[UNK]"]
        return [self._encoding_mapping.get(token, unk_token_id) for token in raw_tokens]

    @override
    def decode(self, encoding: list[int]) -> str:
        return " ".join(self._decoding_mapping.get(idx, "?") for idx in encoding)

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    @override
    def vocabulary(self) -> list[str]:
        return self._vocabulary

    @override
    def save(self, tokenizer_path: Path) -> None:
        """Saves the vocabulary and config file."""
        super().save(tokenizer_path)
        vocab_file = tokenizer_path / "vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, indent=2)

        config = {
            "tokenizer_type": "ChessTokenizer",
            "vocab_file": "vocab.json",
        }
        config_path = tokenizer_path / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    @override
    def load(cls, tokenizer_path: Path) -> Self:
        """Loads a ChessTokenizer from a directory."""
        config_path = tokenizer_path / "tokenizer_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        vocab_filename = config.get("vocab_file")
        if not vocab_filename:
            raise ValueError("Tokenizer config is missing 'vocab_file' key.")

        vocab_file = tokenizer_path / vocab_filename
        if not vocab_file.is_file():
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")

        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)

        return cls(vocab=vocab)


def main() -> None:
    tokenizer = ChessTokenizer()
    game = """
1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e6 6.f4 a6 7.Qf3 Qb6 8.Nb3 Qc7
9.Bd3 b5 10.g4 b4 11.Ne2 Bb7 12.g5 Nfd7 13.Bd2 Nc6 14.Nbd4 Nc5 15.Nxc6 Qxc6
16.Nd4 Qd7 17.O-O-O Qa4 18.Kb1 b3 19.Nxb3 Nxe4 20.Qf1 g6 21.Be1 Bg7 22.h4 O-O
23.h5 Nc5 24.Rh4 Nxd3 25.Rxd3 Be4 26.hxg6 fxg6 27.Rc3 Rac8 28.Rh2 Bf5 29.Qf2 Bxc3
30.Bxc3 e5 31.fxe5 Bd3 32.Qe3 Rf1+ 33.Nc1 Rxc3 34.bxc3 Qb5+ 35.Ka1 Bc4 36.exd6 Bf7
37.Rf2 Rxf2 38.Qxf2 Qxg5 39.Kb2 h5 40.Qd4 h4 41.Nd3 h3 42.d7 Qd8 43.Ne5 h2
44.Nc6 Qxd7 45.Qxd7 h1=Q 46.Ne5 Qf1 47.Nxf7 Qxf7 48.Qc8+ Kg7 49.Qxa6 g5 50.a4 g4
51.Qb5 Qg6 52.a5 g3 53.a6 g2 54.Qb7+ Kh6 55.a7 g1=Q 56.a8=Q Q6b6+ 57.Qxb6+ Qxb6+
58.Ka2 Qe6+ 59.Ka1 Qc4 60.Qf8+ Kg6 61.Qb4 Qf1+ 62.Kb2 Kh7 63.Qe4+ Kg8 64.Qd5+ Kh8
65.Qh5+ Kg8 66.Qg5+ Kh8 67.Qd8+ Kh7 68.Qd7+ Kg8 69.Qe8+ Kg7 70.Qe7+ Kg8 71.Qd8+ Kh7
72.Qd7+ Kh8 73.Qd4+ Kg8 74.c4 Qf8 75.Qd5+ Kh8 76.Qe5+ Kh7 77.Qh5+ Kg8 78.Qd5+ Kh7
79.c5 Qb8+ 80.Kc3 Kh8 81.Qd4+ Kg8 82.c6  1-0""".strip().replace("\n", " ")

    print(f"{tokenizer.vocab_size=}")
    print(f"{game=}")
    tokens = tokenizer.encode(game)
    print(f"{tokens[:10]=}")

    decoded_game = tokenizer.decode(tokens)
    print(f"{decoded_game=}")


if __name__ == "__main__":
    main()
