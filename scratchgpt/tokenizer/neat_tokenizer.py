import decimal
import re
from typing import override # Assuming Python 3.12+ or typing_extensions

# Assuming this file is scratchgpt/tokenizer/neat_genome_tokenizer.py
# And base_tokenizer.py is in the same directory.
from .base_tokenizer import Tokenizer

ACTIVATION_FNS: list[str] = list("abs clamped cube exp gauss hat identity inv log relu sigmoid sin softplus square tanh".split())
AGGREGATION_FNS: list[str] = list("sum product mean min max median maxabs".split())
OPS_NAMES: list[str] = ["</xor>", "</xnor>", "</a_nimp_b>", "</b_nimp_a>"]
# Added connection_gene_type based on example data
GENE_TYPE_STRINGS: list[str] = ["DefaultNodeGene", "DefaultConnectionGene"]


class NeatGenomeTokenizer(Tokenizer):
    """
    Tokenizer for serialized NEAT genome strings.

    Converts genome strings into sequences of integer tokens and back.
    The vocabulary is built based on predefined markers, keywords, user-provided lists
    for operations/activations/aggregations, and generated numerical strings within
    specified limits.
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
        Initializes the NeatGenomeTokenizer and builds its vocabulary.
        Args:
            float_min_val: Minimum value for floats (weights, biases, responses).
            float_max_val: Maximum value for floats.
            float_precision: Number of decimal places for float string representations.
            max_inputs: Max inputs for vocabulary generation and validation (1 to N).
            max_outputs: Max outputs for vocabulary generation and validation (1 to M).
            max_hidden_nodes_in_genome: Max hidden nodes allowed IN A GENOME for validation.
                                        Vocabulary built to cover IDs up to (max_outputs-1) + max_hidden_nodes_in_genome.
            start_token_id: The starting integer ID for tokens.
        """
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        self._current_token_id = start_token_id
        self.float_precision = float_precision

        self._max_inputs_limit = max_inputs
        self._max_outputs_limit = max_outputs
        self._max_hidden_nodes_in_genome_limit = max_hidden_nodes_in_genome

        def add_token(token_str: str):
            if token_str not in self._stoi:
                self._stoi[token_str] = self._current_token_id
                self._itos[self._current_token_id] = token_str
                self._current_token_id += 1

        core_tokens = [
            "_S_GEN_", "_E_GEN_", "_S_CONF_", "_E_CONF_", "_S_NODES_", "_E_NODES_",
            "_S_CONNS_", "_E_CONNS_", "_NODE_", "_CONN_",
            "num_inputs", "num_outputs", "feed_forward", "node_gene_type", "connection_gene_type",
            "|", "True", "False"
        ]
        for token in core_tokens:
            add_token(token)
        for token in GENE_TYPE_STRINGS: # Using the global GENE_TYPE_STRINGS
            add_token(token)

        for token_list in [OPS_NAMES, ACTIVATION_FNS, AGGREGATION_FNS]:
            for token in token_list:
                add_token(token)

        d_ctx = decimal.Context()
        d_ctx.prec = float_precision + 10

        d_float_min_val = decimal.Decimal(str(float_min_val), d_ctx)
        d_float_max_val = decimal.Decimal(str(float_max_val), d_ctx)

        if d_float_min_val > d_float_max_val:
            raise ValueError("float_min_val cannot be greater than float_max_val.")

        d_step = decimal.Decimal(f"1e-{float_precision}", d_ctx)

        if d_float_min_val == d_float_max_val:
            add_token(f"{d_float_min_val:.{float_precision}f}")
        else:
            num_discrete_steps = int(((d_float_max_val - d_float_min_val) / d_step).to_integral_value(rounding=decimal.ROUND_HALF_UP))
            for i in range(num_discrete_steps + 1):
                current_d_val = d_float_min_val + i * d_step
                if current_d_val > d_float_max_val: # Clamp
                    current_d_val = d_float_max_val
                add_token(f"{current_d_val:.{float_precision}f}")
            max_val_formatted_str = f"{d_float_max_val:.{float_precision}f}"
            if max_val_formatted_str not in self._stoi:
                 add_token(max_val_formatted_str)

        for i in range(1, self._max_inputs_limit + 1):
            add_token(f"{-i}")

        max_positive_node_id_val = (self._max_outputs_limit + self._max_hidden_nodes_in_genome_limit - 1)
        max_positive_node_id_val = max(self._max_outputs_limit - 1, max_positive_node_id_val)
        max_positive_node_id_val = max(0, max_positive_node_id_val)

        for i in range(max_positive_node_id_val + 1):
            add_token(f"{i}")

        for i in range(1, self._max_inputs_limit + 1): # For N values
            add_token(f"{i}")
        for i in range(1, self._max_outputs_limit + 1): # For M values
            add_token(f"{i}")

        self._sorted_vocab_keys = sorted(self._stoi.keys(), key=len, reverse=True)
        self._vocab_list = [""] * (self._current_token_id - start_token_id)
        for s, i_val in self._stoi.items():
            self._vocab_list[i_val - start_token_id] = s

    def _validate_genome_structure(self, text: str) -> tuple[int, int]: # Return N, M for use
        n_val = -1
        m_val = -1
        try:
            conf_match = re.search(r"_S_CONF_(.*?)_E_CONF_", text)
            if not conf_match:
                raise ValueError("Could not find _S_CONF_..._E_CONF_ block.")
            conf_content = conf_match.group(1).strip()
            conf_parts = conf_content.split('|')

            for i in range(len(conf_parts) -1): # Iterate up to second to last for key-value pairs
                if conf_parts[i] == "num_inputs" and (i + 1) < len(conf_parts):
                    n_val = int(conf_parts[i+1])
                elif conf_parts[i] == "num_outputs" and (i + 1) < len(conf_parts):
                    m_val = int(conf_parts[i+1])

            if n_val == -1: raise ValueError("num_inputs not found or malformed.")
            if m_val == -1: raise ValueError("num_outputs not found or malformed.")

            if not (1 <= n_val <= self._max_inputs_limit):
                raise ValueError(f"Number of inputs N={n_val} out of range [1, {self._max_inputs_limit}].")
            if not (1 <= m_val <= self._max_outputs_limit):
                raise ValueError(f"Number of outputs M={m_val} out of range [1, {self._max_outputs_limit}].")
        except Exception as e:
            raise ValueError(f"Error parsing/validating _S_CONF_ block: {e}")

        try:
            # Determine search boundary for nodes_content more reliably
            nodes_section_start_idx = text.find("_S_NODES_")
            if nodes_section_start_idx == -1:
                raise ValueError("Could not find start of _S_NODES_ block.")

            s_conns_idx = text.find("_S_CONNS_", nodes_section_start_idx)
            e_gen_idx = text.find("_E_GEN_", nodes_section_start_idx)

            if s_conns_idx != -1 and (e_gen_idx == -1 or s_conns_idx < e_gen_idx):
                nodes_section_end_idx = s_conns_idx
            elif e_gen_idx != -1:
                nodes_section_end_idx = e_gen_idx
            else:
                raise ValueError("Could not determine end of _S_NODES_ block.")

            nodes_content_full = text[nodes_section_start_idx + len("_S_NODES_") : nodes_section_end_idx]

            hidden_node_ids_found: set[int] = set()
            for match in re.finditer(r"_NODE_([^|]+?)\|.*?_E_NODES_", nodes_content_full):
                node_id_str = match.group(1)
                try:
                    node_id = int(node_id_str)
                    if node_id >= m_val:
                        hidden_node_ids_found.add(node_id)
                except ValueError:
                    raise ValueError(f"Invalid node_id found: {node_id_str}")

            num_hidden_nodes = len(hidden_node_ids_found)
            if num_hidden_nodes > self._max_hidden_nodes_in_genome_limit:
                raise ValueError(f"Num hidden nodes {num_hidden_nodes} exceeds limit {self._max_hidden_nodes_in_genome_limit}.")
        except Exception as e:
            raise ValueError(f"Error parsing/validating _S_NODES_ or hidden count: {e}")
        return n_val, m_val


    @override
    def encode(self, text: str) -> list[int]:
        self._validate_genome_structure(text)

        encoded_tokens: list[int] = []
        current_pos = 0
        text_len = len(text)

        while current_pos < text_len:
            found_match = False
            for token_str in self._sorted_vocab_keys:
                if text.startswith(token_str, current_pos):
                    encoded_tokens.append(self._stoi[token_str])
                    current_pos += len(token_str)
                    found_match = True
                    break
            if not found_match:
                raise ValueError(
                    f"Untokenizable sequence at pos {current_pos}. "
                    f"Snippet: '{text[current_pos:current_pos+20]}...'. "
                    f"Ensure numbers formatted to {self.float_precision} decimals, "
                    f"within ranges, and IDs within limits."
                )
        return encoded_tokens

    @override
    def decode(self, encoding: list[int]) -> str:
        return "".join([self._itos[token_id] for token_id in encoding])

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._stoi)

    @property
    @override
    def vocabulary(self) -> list[str]:
        return self._vocab_list