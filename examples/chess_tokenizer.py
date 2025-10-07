import json
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
        tokens = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "*", "+", "#"}
        tokens.add("O-O")  # Kingside castling
        tokens.add("O-O-O")  # Queenside castling

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
        # Add spaces around check/mate symbols to ensure they are tokenized separately
        processed_text = text.replace("+", " + ").replace("#", " # ")

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
1. e4 d5 2. Nf3 dxe4 3. Ng5 Bf5 4. Nc3 Qd4 5. Qe2 Nf6 6. Qb5+ Nbd7 7. Qxb7 Rb8 8. Qxc7 h6 9. Nh3 Bxh3 10. gxh3 e6
11. Qg3 Ne5 12. Bb5+ Ke7 13. Be2 g5 14. O-O h5 15. d3 g4 16. Qh4 gxh3 17. Bg5 Kd6 18. Bxf6 Rg8+ 19. Kh1 Ng4
20. Bxd4 Rxb2 21. Qg3+ e5 22. Nxe4+ Kd5 23. Nf6+ Nxf6 24. Qxe5+ Kc6 25. Qxf6+ Bd6 26. Bxb2 Rg2 27. Qf3+ Kc5 28. d4+ Kb4
29. c3+ Ka4 30. Bd1+ Kb5 31. a4+ Kc4 32. Qd3+ Kd5 33. Bf3+ Ke6 34. Bxg2 hxg2+ 35. Kxg2 h4 36. Qe4+ Kd7 37. f4 f5
38. Qxf5+ Kc6 39. d5+ Kc5 40. Qd3 Kb6 41. Qd4+ Bc5 42. Qc4 a5 43. Ba3 Bxa3 44. Qb5+ Kc7 45. Rxa3 Kd6 46. c4 Ke7
47. Qc6 Kf7 48. Re3
""".strip().replace("\n", " ")

    print(f"{tokenizer.vocab_size=}")
    print(f"{game=}")
    tokens = tokenizer.encode(game)
    print(f"{tokens[:10]=}")

    decoded_game = tokenizer.decode(tokens)
    print(f"{decoded_game=}")


if __name__ == "__main__":
    main()
