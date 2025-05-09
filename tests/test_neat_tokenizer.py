import re
import unittest
# Assuming the tokenizer is in: scratchgpt.tokenizer.neat_genome_tokenizer
from scratchgpt.tokenizer.neat_tokenizer import NeatGenomeTokenizer

raw_test_data_genome = """_S_GEN_ </xnor>
_S_CONF_ num_inputs 2 num_outputs 1 feed_forward True node_gene_type DefaultNodeGene connection_gene_type DefaultConnectionGene _E_CONF_
_S_NODES_
_NODE_ 0 -0.04 1.00 sigmoid product _E_NODES_
_NODE_ 1 0.44 1.00 log sum _E_NODES_
_NODE_ 2 -0.63 1.00 log sum _E_NODES_
_NODE_ 3 -0.81 1.00 inv product _E_NODES_
_NODE_ 4 -0.10 1.00 log max _E_NODES_
_S_CONNS_
_CONN_ -2 4 1.54 _E_CONNS_
_CONN_ -1 2 -2.65 _E_CONNS_
_CONN_ -1 3 1.90 _E_CONNS_
_CONN_ 0 2 -1.05 _E_CONNS_
_CONN_ 3 0 0.54 _E_CONNS_
_CONN_ 4 0 3.00 _E_CONNS_
_E_GEN_
"""

def preprocess_genome_string(raw_genome_str: str) -> str:
    # Fix concatenated tokens before _E_NODES_ or _E_CONNS_
    # This specifically targets known patterns from the example.
    # A more general approach might be needed if other tokens can be concatenated.
    processed_str = raw_genome_str
    # Patterns like "token_E_NODES_" -> "token _E_NODES_"
    # and "token_E_CONNS_" -> "token _E_CONNS_"
    # This regex identifies a non-whitespace char sequence followed by _E_...
    # then inserts a space if the first part isn't already a space.
    processed_str = re.sub(r'([^\s])(_E_NODES_)', r'\1 \2', processed_str)
    processed_str = re.sub(r'([^\s])(_E_CONNS_)', r'\1 \2', processed_str)

    processed_str = processed_str.replace('\r\n', ' ').replace('\n', ' ')
    processed_str = re.sub(r' +', ' ', processed_str) # Normalize multiple spaces
    return processed_str.strip()

# Preprocess the sample genome for testing
# This is crucial for the sample to be valid with the strict space separation rule
PROCESSED_TEST_GENOME = preprocess_genome_string(raw_test_data_genome)

class TestNeatGenomeTokenizerSetup(unittest.TestCase):
    def test_invalid_precision(self):
        with self.assertRaisesRegex(ValueError, "float_precision must be between 0 and 15."):
            NeatGenomeTokenizer(0.0, 1.0, -1)
        with self.assertRaisesRegex(ValueError, "float_precision must be between 0 and 15."):
            NeatGenomeTokenizer(0.0, 1.0, 16)

    def test_invalid_float_range(self):
        with self.assertRaisesRegex(ValueError, "float_min_val cannot be greater than float_max_val."):
            NeatGenomeTokenizer(1.0, 0.0, 2)

    def test_invalid_max_inputs(self):
        with self.assertRaisesRegex(ValueError, "max_inputs must be positive."):
            NeatGenomeTokenizer(0.0, 1.0, 2, max_inputs=0)
        with self.assertRaisesRegex(ValueError, "max_inputs must be positive."):
            NeatGenomeTokenizer(0.0, 1.0, 2, max_inputs=-1)

    def test_invalid_max_outputs(self):
        with self.assertRaisesRegex(ValueError, "max_outputs must be positive."):
            NeatGenomeTokenizer(0.0, 1.0, 2, max_outputs=0)

    def test_invalid_max_hidden_nodes(self):
        with self.assertRaisesRegex(ValueError, "max_hidden_nodes_in_genome must be non-negative."):
            NeatGenomeTokenizer(0.0, 1.0, 2, max_hidden_nodes_in_genome=-1)

    def test_valid_minimal_params(self):
        try:
            NeatGenomeTokenizer(0.0, 0.0, 0, max_inputs=1, max_outputs=1, max_hidden_nodes_in_genome=0)
        except ValueError:
            self.fail("NeatGenomeTokenizer raised ValueError unexpectedly with minimal valid params.")


class TestNeatGenomeTokenizerOperations(unittest.TestCase):
    def setUp(self):
        # Configured to handle the PROCESSED_TEST_GENOME
        self.tokenizer = NeatGenomeTokenizer(
            float_min_val=-3.0,
            float_max_val=3.0,
            float_precision=2,
            max_inputs=2, # From "num_inputs 2", IDs -1, -2
            max_outputs=1, # From "num_outputs 1", ID 0
            max_hidden_nodes_in_genome=4, # Node IDs 0,1,2,3,4 used. Max is 4. 1 output + 4 hidden = 5 nodes (0-4)
            start_token_id=0
        )
        self.test_genome_str = PROCESSED_TEST_GENOME
        # Expected tokens in PROCESSED_TEST_GENOME:
        # Floats: -0.04, 1.00, 0.44, -0.63, -0.81, -0.10, 1.54, -2.65, 1.90, -1.05, 0.54, 3.00
        # Ints: 0, 1, 2, 3, 4, -1, -2
        # OpName: </xnor>
        # Keywords, Structurals, Bools, GeneTypes, Activations, Aggregations, Space

    def test_encode_decode_happy_path(self):
        encoded_ids = self.tokenizer.encode(self.test_genome_str)
        self.assertIsInstance(encoded_ids, list)
        self.assertTrue(all(isinstance(id_val, int) for id_val in encoded_ids))

        decoded_str = self.tokenizer.decode(encoded_ids)
        self.assertEqual(decoded_str, self.test_genome_str)

    def test_vocab_properties(self):
        self.assertGreater(self.tokenizer.vocab_size, 0)
        vocab_list = self.tokenizer.vocabulary
        self.assertIsInstance(vocab_list, list)
        self.assertEqual(len(vocab_list), self.tokenizer.vocab_size)
        self.assertTrue(all(isinstance(token, str) for token in vocab_list))
        # Check if a few key tokens are present
        self.assertIn("_S_GEN_", vocab_list)
        self.assertIn(" ", vocab_list) # Space token
        self.assertIn("0.50", vocab_list) # Example float if within range -3.0 to 3.0
        self.assertIn("</xnor>", vocab_list) # From DEFAULT_OP_NAMES

    def test_encode_empty_string(self):
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_decode_empty_list(self):
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_start_token_id_offset(self):
        tokenizer_offset = NeatGenomeTokenizer(
            float_min_val=0.0, float_max_val=0.1, float_precision=1,
            start_token_id=100
        )
        encoded = tokenizer_offset.encode("0.0")
        self.assertTrue(all(id_val >= 100 for id_val in encoded))
        # Test that first token gets start_token_id
        first_token_str = tokenizer_offset.vocabulary[0] # This is _S_GEN_
        expected_first_id = tokenizer_offset.start_token_id
        self.assertEqual(tokenizer_offset._stoi[first_token_str], expected_first_id)


    def test_float_precision_zero(self):
        tokenizer_prec0 = NeatGenomeTokenizer(
            float_min_val=-2.0, float_max_val=2.0, float_precision=0,
            max_inputs=1, max_outputs=1, max_hidden_nodes_in_genome=0
        )
        self.assertIn("-2", tokenizer_prec0.vocabulary) # from -2.0
        self.assertIn("0", tokenizer_prec0.vocabulary)  # from 0.0 and also an ID
        self.assertIn("1", tokenizer_prec0.vocabulary)  # from 1.0 and also an ID/count
        encoded = tokenizer_prec0.encode("-1 True 0")
        decoded = tokenizer_prec0.decode(encoded)
        self.assertEqual(decoded, "-1 True 0")

    def test_float_min_equals_max(self):
        tokenizer_min_max_eq = NeatGenomeTokenizer(
            float_min_val=1.23, float_max_val=1.23, float_precision=2,
        )
        self.assertIn("1.23", tokenizer_min_max_eq.vocabulary)
        # Check that only one float related to this specific setup is there if range is tight
        float_tokens = [t for t in tokenizer_min_max_eq.vocabulary if re.match(r"^-?\d+\.\d+$", t)]
        self.assertIn("1.23", float_tokens)
        # Depending on other float settings, there might be more than one.
        # This test ensures "1.23" is present.

    def test_encode_unknown_token(self):
        with self.assertRaisesRegex(ValueError, "Token 'UNKNOWN_TOKEN' not found in vocabulary"):
            self.tokenizer.encode("_S_GEN_ UNKNOWN_TOKEN _E_GEN_")

    def test_decode_unknown_token_id(self):
        max_id = self.tokenizer.vocab_size + self.tokenizer.start_token_id -1
        unknown_id = max_id + 1
        with self.assertRaisesRegex(ValueError, f"Token ID '{unknown_id}' not found in vocabulary"):
            self.tokenizer.decode([self.tokenizer._stoi["_S_GEN_"], unknown_id])

    def test_encode_type_error(self):
        with self.assertRaisesRegex(TypeError, "Input text must be a string."):
            self.tokenizer.encode(123) # type: ignore

    def test_decode_type_error(self):
        with self.assertRaisesRegex(TypeError, "Input encoding must be a list of integers."):
            self.tokenizer.decode("not a list") # type: ignore
        with self.assertRaisesRegex(ValueError, "All items in encoding list must be integers."):
            self.tokenizer.decode([0, 1, "two"]) # type: ignore

    def test_string_with_multiple_spaces(self):
        # "A  B" should be tokenized as "A", " ", " ", "B"
        # based on re.split(r'( )', text) and subsequent filtering of empty strings.
        text = "_S_GEN_  _E_GEN_" # Two spaces
        expected_tokens = ["_S_GEN_", " ", " ", "_E_GEN_"]

        encoded = self.tokenizer.encode(text)

        # Verify encoded IDs map back to the expected sequence of token strings
        actual_tokens_from_ids = [self.tokenizer._itos[token_id] for token_id in encoded]
        self.assertEqual(actual_tokens_from_ids, expected_tokens)

        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_tokens_are_unique_and_ids_are_correct(self):
        # Check if all tokens in vocab are unique
        self.assertEqual(len(self.tokenizer.vocabulary), len(set(self.tokenizer.vocabulary)))
        # Check if _stoi and _itos are consistent and cover the vocab
        self.assertEqual(len(self.tokenizer._stoi), self.tokenizer.vocab_size)
        self.assertEqual(len(self.tokenizer._itos), self.tokenizer.vocab_size)
        for i, token_str in enumerate(self.tokenizer.vocabulary):
            expected_id = self.tokenizer.start_token_id + i
            self.assertEqual(self.tokenizer._stoi[token_str], expected_id)
            self.assertEqual(self.tokenizer._itos[expected_id], token_str)

    def test_zero_float_representation(self):
        # Test specific representation of 0.0, -0.0 etc.
        tok_zero_prec = NeatGenomeTokenizer(float_min_val=-0.5, float_max_val=0.5, float_precision=2)
        self.assertIn("0.00", tok_zero_prec.vocabulary)
        self.assertNotIn("-0.00", tok_zero_prec.vocabulary) # Should be normalized
        self.assertIn("0.10", tok_zero_prec.vocabulary)
        self.assertIn("-0.10", tok_zero_prec.vocabulary)

        tok_zero_prec0 = NeatGenomeTokenizer(float_min_val=-0.0, float_max_val=0.0, float_precision=0)
        self.assertIn("0", tok_zero_prec0.vocabulary) # "0" for 0.0 with precision 0
        self.assertNotIn("-0", tok_zero_prec0.vocabulary)


if __name__ == '__main__':
    print(f"Processed Test Genome for testing:\n'{PROCESSED_TEST_GENOME}'\n")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
