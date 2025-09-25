import json
from pathlib import Path

import pytest

from examples.chess_tokenizer import ChessTokenizer
from scratchgpt.model_io import save_tokenizer


def test_save_and_load_happy_path(tmp_path: Path):
    """Tests standard saving and loading of a ChessTokenizer."""
    original_tokenizer = ChessTokenizer()
    tokenizer_dir = tmp_path / "experiment"

    save_tokenizer(tokenizer_dir, original_tokenizer)

    # Use the class's own .load() method for a direct unit test
    loaded_tokenizer = ChessTokenizer.load(tokenizer_dir / "tokenizer")

    assert isinstance(loaded_tokenizer, ChessTokenizer)
    assert loaded_tokenizer.vocabulary == original_tokenizer.vocabulary
    assert loaded_tokenizer.vocab_size == original_tokenizer.vocab_size


def test_chess_move_encoding_and_decoding():
    """Tests encoding and decoding of various chess moves."""
    tokenizer = ChessTokenizer()

    # Test basic moves
    basic_moves = "1. e4 e5 2. Nf3 Nc6"
    encoded = tokenizer.encode(basic_moves)
    decoded = tokenizer.decode(encoded)
    assert decoded == basic_moves

    # Test captures
    captures = "1. e4 d5 2. exd5 Qxd5"
    encoded = tokenizer.encode(captures)
    decoded = tokenizer.decode(encoded)
    assert decoded == captures

    # Test check and checkmate
    check_mate = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7+ Ke7 5. Qf3#"
    # The tokenizer should handle + and # by adding spaces around them
    encoded = tokenizer.encode(check_mate)
    decoded = tokenizer.decode(encoded)
    expected = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7 + Ke7 5. Qf3 #"
    assert decoded == expected


def test_castling_moves():
    """Tests that castling moves are properly tokenized."""
    tokenizer = ChessTokenizer()

    # These should be in the vocabulary for the tokenizer to work properly
    # If they're not, the test will fail and indicate what needs to be fixed
    castling_game = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O"

    encoded = tokenizer.encode(castling_game)
    decoded = tokenizer.decode(encoded)

    # Check that no [UNK] tokens were generated
    assert "[UNK]" not in decoded

    # The moves should decode properly (castling might have spaces added around them)
    # This test will reveal if castling moves are missing from vocabulary
    tokens = decoded.split()
    assert "O-O" in tokens or "O - O" in tokens  # Either form should work


def test_promotion_moves():
    """Tests pawn promotion moves."""
    tokenizer = ChessTokenizer()

    promotion_moves = """
1. e4 e5 2. d4 exd4 3. c3 dxc3 4. bxc3 d6 5. Nf3 Bg4 6. Be2 Nc6 7. O-O Qd7 8. Re1 O-O-O 9. Nd4 Bxe2 10. Qxe2 Nxd4
11. cxd4 Kb8 12. Nc3 f5 13. exf5 Qxf5 14. Qe8+ Rxe8 15. Rxe8+ Kc7 16. d5 Qxf2+ 17. Kh1 Qf1+ 18. Bg1 Ne7 19. Rae1 Nxd5
20. Nxd5+ Kd7 21. Re7+ Kd8 22. Rxg7 Qxe1 23. Rxg1 Qe2 24. Rg8+ Ke8 25. Nf6+ Kf7 26. Nh5+ Kf8 27. Rg3 Qe1
28. Rf3+ Kg8 29. Nf6+ Kh8 30. Nxh7 Qe8+ 31. Rf8 Qxf8+ 32. Nxf8 a5 33. Nd7 a4 34. Nxc5 dxc5 35. a3 c4 36. Kg2 c3
37. Kf3 c2 38. Ke2 c1=Q""".strip().replace("\n", " ")

    encoded = tokenizer.encode(promotion_moves)
    decoded = tokenizer.decode(encoded)

    # Should not have unknown tokens
    assert "[UNK]" not in decoded

    # Should contain promotion notation
    assert "c1=Q" in decoded or "c1 = Q" in decoded


def test_unknown_token_handling():
    """Tests handling of unknown tokens."""
    tokenizer = ChessTokenizer()

    # Test with some invalid chess notation
    invalid_moves = "1. e4 xyz 2. Nf3 invalid_move"
    encoded = tokenizer.encode(invalid_moves)
    decoded = tokenizer.decode(encoded)

    # Unknown tokens should be replaced with [UNK] in encoding
    unk_id = tokenizer._encoding_mapping["[UNK]"]
    assert unk_id in encoded

    # Decoding should show the [UNK] token for unknown moves
    assert "[UNK]" in decoded


def test_special_tokens():
    """Tests that special tokens are in vocabulary."""
    tokenizer = ChessTokenizer()
    vocab = tokenizer.vocabulary

    # Check that special tokens exist
    expected_special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "*", "+", "#"]
    for token in expected_special:
        assert token in vocab, f"Special token {token} missing from vocabulary"


def test_vocabulary_size():
    """Tests vocabulary size is reasonable."""
    tokenizer = ChessTokenizer()
    vocab_size = tokenizer.vocab_size

    # Chess vocabulary should be substantial but not excessive
    # This is a sanity check - adjust bounds if needed
    assert 1000 < vocab_size < 50000, f"Vocabulary size {vocab_size} seems unreasonable"

    # All vocabulary items should be unique
    assert len(tokenizer.vocabulary) == len(set(tokenizer.vocabulary))


def test_save_and_load_preserves_functionality(tmp_path: Path):
    """Tests that saved and loaded tokenizer works identically."""
    original_tokenizer = ChessTokenizer()
    tokenizer_dir = tmp_path / "experiment"

    # Test game from the original example
    test_game = "1. e4 d5 2. Nf3 dxe4 3. Ng5 Bf5 4. Nc3 Qd4"

    # Encode with original
    original_encoded = original_tokenizer.encode(test_game)
    original_decoded = original_tokenizer.decode(original_encoded)

    # Save and load
    save_tokenizer(tokenizer_dir, original_tokenizer)
    loaded_tokenizer = ChessTokenizer.load(tokenizer_dir / "tokenizer")

    # Encode with loaded tokenizer
    loaded_encoded = loaded_tokenizer.encode(test_game)
    loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

    # Should be identical
    assert original_encoded == loaded_encoded
    assert original_decoded == loaded_decoded


def test_load_error_missing_vocab_file(tmp_path: Path):
    """Tests that ChessTokenizer.load() fails if vocab.json is missing."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    # Manually create only the config file, but not the vocab file
    config = {"tokenizer_type": "ChessTokenizer", "vocab_file": "vocab.json"}
    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f)

    with pytest.raises(FileNotFoundError, match="Vocabulary file not found"):
        ChessTokenizer.load(tokenizer_dir)


def test_load_error_missing_config_file(tmp_path: Path):
    """Tests that ChessTokenizer.load() fails if config file is missing."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Tokenizer config not found"):
        ChessTokenizer.load(tokenizer_dir)


def test_load_error_malformed_config(tmp_path: Path):
    """Tests that ChessTokenizer.load() fails if config is malformed."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    # Create config without vocab_file key
    config = {"tokenizer_type": "ChessTokenizer"}
    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f)

    with pytest.raises(ValueError, match="missing 'vocab_file' key"):
        ChessTokenizer.load(tokenizer_dir)


def test_real_chess_game_compatibility():
    """Tests with a real chess game similar to what preprocessing would produce."""
    tokenizer = ChessTokenizer()

    # This is similar to what our chess.py preprocessing would produce
    real_game = """
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6
12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe5 18. Nxe5 dxe5 19. Bxf6 Bxf6 20. Nd2 c4
21. Bc2 Qc7 22. Ne4 Be7 23. Qd4 Rfd8 24. Qxe5 Qxe5 25. Nxe5
""".strip().replace("\n", " ")

    encoded = tokenizer.encode(real_game)
    decoded = tokenizer.decode(encoded)

    # Should encode without unknown tokens
    unk_count = sum(1 for token_id in encoded if tokenizer._decoding_mapping.get(token_id) == "[UNK]")
    assert unk_count == 0, f"Found {unk_count} unknown tokens in real game"

    # Should be able to round-trip
    assert len(encoded) > 0
    assert len(decoded) > 0
