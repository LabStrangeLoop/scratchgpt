import decimal
import re

from .base_tokenizer import Tokenizer

# Global variable for Operation Names as per user clarification
# These are example ops from the spec.
DEFAULT_OP_NAMES: list[str] = ["</xor>", "</xnor>", "</a_nimp_b>", "</b_nimp_a>"]

class NeatGenomeTokenizer(Tokenizer):
    """
    Tokenizer for NEAT (NeuroEvolution of Augmenting Topologies) genomes.
    Converts NEAT genome strings into sequences of integer IDs and vice-versa.
    """

    def __init__(self,
                 float_min_val: float,
                 float_max_val: float,
                 float_precision: int,
                 max_inputs: int = 10,
                 max_outputs: int = 10,
                 max_hidden_nodes_in_genome: int = 30,
                 start_token_id: int = 0):
        """
        Initializes the NEAT Genome Tokenizer.

        Args:
            float_min_val: Minimum value for float tokens.
            float_max_val: Maximum value for float tokens.
            float_precision: Number of decimal places for float tokens.
            max_inputs: Maximum number of input nodes (for generating input ID tokens and N value tokens).
            max_outputs: Maximum number of output nodes (for generating output ID tokens and M value tokens).
            max_hidden_nodes_in_genome: Maximum number of hidden nodes (for generating hidden node ID tokens).
            start_token_id: The starting ID for the first token in the vocabulary.
        """
        if not (0 <= float_precision <= 15): # Practical limit for precision
            raise ValueError("float_precision must be between 0 and 15.")
        if float_min_val > float_max_val:
            raise ValueError("float_min_val cannot be greater than float_max_val.")
        if max_inputs <= 0:
            raise ValueError("max_inputs must be positive.")
        if max_outputs <= 0:
            raise ValueError("max_outputs must be positive.")
        if max_hidden_nodes_in_genome < 0: # Can be 0 if no hidden nodes allowed apart from outputs
             raise ValueError("max_hidden_nodes_in_genome must be non-negative.")


        self.float_min_val = float_min_val
        self.float_max_val = float_max_val
        self.float_precision = float_precision
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        self.max_hidden_nodes_in_genome = max_hidden_nodes_in_genome
        self.start_token_id = start_token_id

        self._vocab: list[str] = []
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        self._current_token_id: int = self.start_token_id

        self._build_vocabulary()

    def _add_token(self, token_str: str) -> None:
        """Adds a token to the vocabulary if not already present."""
        if token_str not in self._stoi:
            self._vocab.append(token_str)
            self._stoi[token_str] = self._current_token_id
            self._itos[self._current_token_id] = token_str
            self._current_token_id += 1

    def _build_vocabulary(self) -> None:
        """Builds the vocabulary based on the NEAT genome specification."""

        # 1. Structural Markers
        structural_markers = [
            "_S_GEN_", "_E_GEN_", "_S_CONF_", "_E_CONF_", "_S_NODES_",
            "_E_NODES_", "_S_CONNS_", "_E_CONNS_", "_NODE_", "_CONN_"
        ]
        for token in structural_markers:
            self._add_token(token)

        # 2. Keywords
        keywords = [
            "num_inputs", "num_outputs", "feed_forward",
            "node_gene_type", "connection_gene_type"
        ]
        for token in keywords:
            self._add_token(token)

        # 3. Boolean Values
        booleans = ["True", "False"]
        for token in booleans:
            self._add_token(token)

        # 4. Gene Type Names
        # Spec: "DefaultNodeGene", "DefaultConnectionGene" (quotes for string literal in markdown)
        gene_types = ["DefaultNodeGene", "DefaultConnectionGene"]
        for token in gene_types:
            self._add_token(token)

        # 5. Activation Function Names
        activations = ["sigmoid", "relu", "tanh", "inv", "log", "abs", "clamped"]
        for token in activations:
            self._add_token(token)

        # 6. Aggregation Function Names
        aggregations = ["sum", "product", "mean", "max"]
        for token in aggregations:
            self._add_token(token)

        # 7. Separator
        self._add_token(" ")

        # 8. Numerical Value Strings (Floats)
        # Using Decimal for precise float generation and formatting
        # Context precision should be higher than target precision for intermediate calcs
        ctx = decimal.Context(prec=self.float_precision + 10)

        scale_factor = ctx.power(10, self.float_precision) # Decimal(10) ** Decimal(self.float_precision)

        # Determine the range of scaled integers
        # Smallest integer i such that (i / scale_factor) >= float_min_val
        start_num_scaled = int((ctx.create_decimal_from_float(self.float_min_val) * scale_factor).to_integral_value(rounding=decimal.ROUND_CEILING))

        # Largest integer i such that (i / scale_factor) <= float_max_val
        end_num_scaled = int((ctx.create_decimal_from_float(self.float_max_val) * scale_factor).to_integral_value(rounding=decimal.ROUND_FLOOR))

        if start_num_scaled <= end_num_scaled:
            for i in range(start_num_scaled, end_num_scaled + 1):
                val_decimal = ctx.divide(decimal.Decimal(i), scale_factor)

                # Normalize -0.0 to 0.0 for consistent string representation "0.00" vs "-0.00"
                if val_decimal.is_zero() and val_decimal.is_signed():
                    val_decimal = decimal.Decimal("0.0") # Use string to ensure it's treated as exact zero

                token_str = f"{val_decimal:.{self.float_precision}f}"
                self._add_token(token_str)

        # 9. Identifier and Count Strings (Integers as Strings)
        # Input Node IDs: "-max_inputs", ..., "-1"
        for i in range(1, self.max_inputs + 1):
            self._add_token(str(-i))

        # Output and Remapped Hidden Node IDs: "0", ..., "max_outputs + max_hidden_nodes_in_genome - 1"
        # Max ID for this category. If max_outputs=1, max_hidden=0, then max_id = 1+0-1 = 0. Range is 0..0.
        # The number of such nodes is max_outputs + max_hidden_nodes_in_genome.
        # IDs run from 0 to (count - 1).
        upper_bound_output_hidden_id = self.max_outputs + self.max_hidden_nodes_in_genome
        for i in range(upper_bound_output_hidden_id):
            self._add_token(str(i))

        # N (Number of Inputs) Values: "1", ..., "max_inputs"
        for i in range(1, self.max_inputs + 1):
            self._add_token(str(i))

        # M (Number of Outputs) Values: "1", ..., "max_outputs"
        for i in range(1, self.max_outputs + 1):
            self._add_token(str(i))

        # 10. Operation Names (Configurable, from global DEFAULT_OP_NAMES)
        # These are added last as per clarification.
        for token in DEFAULT_OP_NAMES:
            self._add_token(token)

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)

    @property
    def vocabulary(self) -> list[str]:
        """Return the learned vocabulary as a list of token strings."""
        return list(self._vocab) # Return a copy

    def encode(self, text: str) -> list[int]:
        """Convert a NEAT genome string into a sequence of token IDs."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        if not text:
            return []

        # Split by the space character, keeping the space itself as a token.
        # Example: "A B" -> ["A", " ", "B"]
        # Example: "A"   -> ["A"]
        # Example: " A " -> ["", " ", "A", " ", ""] -> filter empty parts
        parts = re.split(r'( )', text)

        token_ids: list[int] = []
        for part in parts:
            if not part: # Filter out empty strings that can result from re.split
                continue

            token_id = self._stoi.get(part)
            if token_id is None:
                raise ValueError(f"Token '{part}' not found in vocabulary during encode.")
            token_ids.append(token_id)
        return token_ids

    def decode(self, encoding: list[int]) -> str:
        """Convert a sequence of token IDs back into a NEAT genome string."""
        if not isinstance(encoding, list):
            raise TypeError("Input encoding must be a list of integers.")
        if not all(isinstance(item, int) for item in encoding):
            raise ValueError("All items in encoding list must be integers.")

        try:
            # Since spaces are tokens, joining the decoded strings directly will reconstruct the original.
            return "".join(self._itos[token_id] for token_id in encoding)
        except KeyError as e:
            # e.args[0] will contain the missing key (token_id)
            raise ValueError(f"Token ID '{e.args[0]}' not found in vocabulary during decode.") from e