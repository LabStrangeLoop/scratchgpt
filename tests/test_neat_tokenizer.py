import pytest # Using pytest for fixture and exception testing
# Assuming the tokenizer is in: scratchgpt.tokenizer.neat_genome_tokenizer
from scratchgpt.tokenizer.neat_tokenizer import NeatGenomeTokenizer, OPS_NAMES, ACTIVATION_FNS, AGGREGATION_FNS

# Test data provided by the user
test_data_genome = """_S_GEN_ </xnor>
_S_CONF_ num_inputs|2|num_outputs|1|feed_forward|True|node_gene_type|DefaultNodeGene|connection_gene_type|DefaultConnectionGene _E_CONF_
_S_NODES_
_NODE_0|-0.04|1.00|sigmoid|product_E_NODES_
_NODE_1|0.44|1.00|log|sum_E_NODES_
_NODE_2|-0.63|1.00|log|sum_E_NODES_
_NODE_3|-0.81|1.00|inv|product_E_NODES_
_NODE_4|-0.10|1.00|log|max_E_NODES_
_S_CONNS_
_CONN_-2|4|1.54_E_CONNS_
_CONN_-1|2|-2.65_E_CONNS_
_CONN_-1|3|1.90_E_CONNS_
_CONN_0|2|-1.05_E_CONNS_
_CONN_3|0|0.54_E_CONNS_
_CONN_4|0|3.00_E_CONNS_
_E_GEN_"""

# Clean up the test data string (remove potential leading/trailing newlines and ensure consistent newlines if any)
test_data_genome = test_data_genome.replace('\r\n', '\n').strip()


@pytest.fixture
def genome_tokenizer_instance():
    """Provides a default NeatGenomeTokenizer instance for tests."""
    # Parameters for the tokenizer, make sure they cover the test_data_genome
    # Floats in test_data_genome: -0.04, 1.00, 0.44, -0.63, -0.81, -0.10, 1.54, -2.65, 1.90, -1.05, 0.54, 3.00
    # Min: -2.65, Max: 3.00. Precision: 2
    return NeatGenomeTokenizer(
        float_min_val=-3.0, # Cover -2.65
        float_max_val=3.0,  # Cover 3.00
        float_precision=2,
        max_inputs=2,       # Test data has num_inputs|2
        max_outputs=1,      # Test data has num_outputs|1
        max_hidden_nodes_in_genome=4 # Test data nodes: 0 (output), 1,2,3,4 (hidden). So 4 hidden.
    )

def test_basic_neat_genome_tokenizer_encode_decode(genome_tokenizer_instance: NeatGenomeTokenizer):
    """Tests basic encoding and decoding of a valid genome string."""
    tokenizer = genome_tokenizer_instance

    encoded = tokenizer.encode(test_data_genome)
    decoded = tokenizer.decode(encoded)

    # For debugging if it fails:
    if test_data_genome != decoded:
        print("Original Length:", len(test_data_genome))
        print("Decoded Length:", len(decoded))
        for i in range(min(len(test_data_genome), len(decoded))):
            if test_data_genome[i] != decoded[i]:
                print(f"Mismatch at index {i}: Original='{test_data_genome[i]}' Decoded='{decoded[i]}'")
                print(f"Original context: ...{test_data_genome[max(0,i-10):i+10]}...")
                print(f"Decoded context:  ...{decoded[max(0,i-10):i+10]}...")
                break
        # print("Original Text:\n", test_data_genome)
        # print("Encoded Tokens:\n", encoded)
        # print("Decoded Text:\n", decoded)


    assert test_data_genome == decoded, "Encode-decode cycle failed for NeatGenomeTokenizer"

def test_neat_genome_tokenizer_vocab_properties(genome_tokenizer_instance: NeatGenomeTokenizer):
    """Tests vocabulary size and content."""
    tokenizer = genome_tokenizer_instance
    assert tokenizer.vocab_size > 0, "Vocab size should be greater than 0"

    vocab_list = tokenizer.vocabulary
    assert isinstance(vocab_list, list), "Vocabulary should be a list"
    assert len(vocab_list) == tokenizer.vocab_size, "Vocabulary list length mismatch with vocab_size"
    if vocab_list: # If not empty
        assert isinstance(vocab_list[0], str), "Vocabulary items should be strings"

    # Check if some expected core tokens are present
    assert "_S_GEN_" in tokenizer._stoi, "Core token _S_GEN_ missing"
    assert "|" in tokenizer._stoi, "Separator token | missing"
    assert "True" in tokenizer._stoi, "Boolean token True missing"
    # Check one from each user-defined list (assuming they are not empty)
    if OPS_NAMES:
         assert OPS_NAMES[0] in tokenizer._stoi, f"Operation token {OPS_NAMES[0]} missing"
    if ACTIVATION_FNS:
        assert ACTIVATION_FNS[0] in tokenizer._stoi, f"Activation token {ACTIVATION_FNS[0]} missing"
    if AGGREGATION_FNS:
        assert AGGREGATION_FNS[0] in tokenizer._stoi, f"Aggregation token {AGGREGATION_FNS[0]} missing"

    # Check a sample float token (if range allows)
    # Using the exact string format from vocab generation
    sample_float_str = f"{genome_tokenizer_instance.float_min_val:.{genome_tokenizer_instance.float_precision}f}"
    assert sample_float_str in tokenizer._stoi, f"Sample float token {sample_float_str} missing"

    # Check a sample ID token
    assert "0" in tokenizer._stoi, "Node ID token '0' missing"
    assert "-1" in tokenizer._stoi, "Input ID token '-1' missing"
    assert "1" in tokenizer._stoi, "N/M value token '1' missing"


def test_validation_num_inputs_exceeded(genome_tokenizer_instance: NeatGenomeTokenizer):
    """Tests validation failure for too many inputs."""
    malformed_genome = test_data_genome.replace("num_inputs|2|", "num_inputs|20|") # Max inputs set to 2 in fixture
    with pytest.raises(ValueError, match=r"Number of inputs N=20 out of range"):
        genome_tokenizer_instance.encode(malformed_genome)

def test_validation_num_outputs_exceeded(genome_tokenizer_instance: NeatGenomeTokenizer):
    """Tests validation failure for too many outputs."""
    malformed_genome = test_data_genome.replace("num_outputs|1|", "num_outputs|5|") # Max outputs set to 1 in fixture
    with pytest.raises(ValueError, match=r"Number of outputs M=5 out of range"):
        genome_tokenizer_instance.encode(malformed_genome)

def test_validation_hidden_nodes_exceeded(genome_tokenizer_instance: NeatGenomeTokenizer):
    """Tests validation failure for too many hidden nodes."""
    # Fixture has max_hidden_nodes_in_genome=4. Test data has 4 hidden nodes (1,2,3,4 for M=1).
    # Add one more hidden node to exceed the limit.
    # Original nodes end at _NODE_4|-0.10|1.00|log|max_E_NODES_
    # Add _NODE_5|0.0|1.0|sigmoid|sum_E_NODES_
    additional_node = "_NODE_5|0.00|1.00|sigmoid|sum_E_NODES_\n"
    malformed_genome = test_data_genome.replace(
        "_NODE_4|-0.10|1.00|log|max_E_NODES_",
        "_NODE_4|-0.10|1.00|log|max_E_NODES_\n" + additional_node
    )
    with pytest.raises(ValueError, match=r"Num hidden nodes 5 exceeds limit 4"):
        genome_tokenizer_instance.encode(malformed_genome)

# def test_untokenizable_sequence(genome_tokenizer_instance: NeatGenomeTokenizer):
#     """Tests failure on an untokenizable sequence (e.g., bad float format or unknown token)."""
#     # Introduce a float with wrong precision that won't be in vocab
#     malformed_genome_bad_float = test_data_genome.replace("1.54", "1.5X3")
#     with pytest.raises(ValueError, match=r"Untokenizable sequence at position"):
#         genome_tokenizer_instance.encode(malformed_genome_bad_float)

#     # Introduce an unknown keyword
#     malformed_genome_unknown_token = test_data_genome.replace("sigmoid", "weird_activation_fn")
#     with pytest.raises(ValueError, match=r"Untokenizable sequence at position"):
#         genome_tokenizer_instance.encode(malformed_genome_unknown_token)

if __name__ == "__main__":
    # This allows running the tests with `python test_neat_genome_tokenizer.py`
    # You might need to adjust paths if scratchgpt is not in PYTHONPATH
    # For pytest, just run `pytest` in the directory containing scratchgpt folder
    # or `pytest path/to/test_neat_genome_tokenizer.py`

    # Example of direct run for debugging:
    print("Running NeatGenomeTokenizer tests manually...")

    # Manually create instance for direct run if needed
    tokenizer_manual = NeatGenomeTokenizer(
        float_min_val=-3.0,
        float_max_val=3.0,
        float_precision=2,
        max_inputs=2,
        max_outputs=1,
        max_hidden_nodes_in_genome=4
    )

    print(f"Tokenizer Vocab Size: {tokenizer_manual.vocab_size}")
    # print(f"Tokenizer Vocabulary Sample: {tokenizer_manual.vocabulary[:20] + tokenizer_manual.vocabulary[-20:]}")

    try:
        test_basic_neat_genome_tokenizer_encode_decode(tokenizer_manual)
        print("test_basic_neat_genome_tokenizer_encode_decode: PASSED")

        test_neat_genome_tokenizer_vocab_properties(tokenizer_manual)
        print("test_neat_genome_tokenizer_vocab_properties: PASSED")

        # For manual testing of exceptions
        print("\nTesting validation errors (expect ValueErrors):")
        try:
            test_validation_num_inputs_exceeded(tokenizer_manual)
        except ValueError as e:
            print(f"  Num inputs exceeded (expected): {e}")

        try:
            test_validation_num_outputs_exceeded(tokenizer_manual)
        except ValueError as e:
            print(f"  Num outputs exceeded (expected): {e}")

        try:
            test_validation_hidden_nodes_exceeded(tokenizer_manual)
        except ValueError as e:
            print(f"  Hidden nodes exceeded (expected): {e}")

        try:
            test_untokenizable_sequence(tokenizer_manual) # This calls it twice, better to split the cases.
        except ValueError as e: # This will only catch the first error from test_untokenizable_sequence
            print(f"  Untokenizable sequence (expected): {e}")

    except Exception as e:
        print(f"A manual test failed: {e}")