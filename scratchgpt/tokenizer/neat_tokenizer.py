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
    # ... (rest of the class is the same as your version initially) ...

    def __init__(self,
                 float_min_val: float,
                 float_max_val: float,
                 float_precision: int,
                 max_inputs: int = 10,
                 max_outputs: int = 10,
                 max_hidden_nodes_in_genome: int = 30,
                 start_token_id: int = 0):
        # ... (initial variable setups are the same) ...
        # gene_type_strings: list[str] = ["DefaultNodeGene"] # This was in your __init__ body, I'll use the global GENE_TYPE_STRINGS
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        self._current_token_id = start_token_id
        self.float_precision = float_precision
        self.float_min_val = float_min_val
        self.float_max_val = float_max_val

        self._max_inputs_limit = max_inputs
        self._max_outputs_limit = max_outputs
        self._max_hidden_nodes_in_genome_limit = max_hidden_nodes_in_genome

        def add_token(token_str: str):
            if token_str not in self._stoi:
                self._stoi[token_str] = self._current_token_id
                self._itos[self._current_token_id] = token_str
                self._current_token_id += 1

        # Group 1: Core Syntax & Keywords & WHITESPACE
        core_tokens = [
            "_S_GEN_", "_E_GEN_", "_S_CONF_", "_E_CONF_", "_S_NODES_", "_E_NODES_",
            "_S_CONNS_", "_E_CONNS_", "_NODE_", "_CONN_",
            "num_inputs", "num_outputs", "feed_forward", "node_gene_type", "connection_gene_type",
            "|", "True", "False"
        ]
        # ---- MODIFICATION START: Add whitespace tokens ----
        whitespace_tokens = [" ", "\n"] # Add other whitespace like "\t" if needed
        # ---- MODIFICATION END ----

        for token in core_tokens:
            add_token(token)

        # ---- MODIFICATION START: Add whitespace tokens to vocab ----
        for token in whitespace_tokens:
            add_token(token)
        # ---- MODIFICATION END ----

        for token in GENE_TYPE_STRINGS: # Using the global GENE_TYPE_STRINGS defined above the class
            add_token(token)

        # Group 2: User-Defined Functional Strings (OPS_NAMES, ACTIVATION_FNS, AGGREGATION_FNS)
        # ... (rest of this group is the same) ...
        for token_list in [OPS_NAMES, ACTIVATION_FNS, AGGREGATION_FNS]:
            for token in token_list:
                add_token(token)

        # Group 3: Numerical Value Strings (Weights, Biases, Responses)
        # ... (rest of this group is the same) ...
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
                if current_d_val > d_float_max_val:
                    current_d_val = d_float_max_val
                add_token(f"{current_d_val:.{float_precision}f}")
            max_val_formatted_str = f"{d_float_max_val:.{float_precision}f}"
            if max_val_formatted_str not in self._stoi:
                 add_token(max_val_formatted_str)

        # Group 4: Identifiers & Counts
        # ... (rest of this group is the same) ...
        for i in range(1, self._max_inputs_limit + 1):
            add_token(f"{-i}")

        max_positive_node_id_val = (self._max_outputs_limit + self._max_hidden_nodes_in_genome_limit - 1)
        max_positive_node_id_val = max(self._max_outputs_limit - 1, max_positive_node_id_val)
        max_positive_node_id_val = max(0, max_positive_node_id_val)

        for i in range(max_positive_node_id_val + 1):
            add_token(f"{i}")

        for i in range(1, self._max_inputs_limit + 1):
            add_token(f"{i}")
        for i in range(1, self._max_outputs_limit + 1):
            add_token(f"{i}")

        # Pre-sort vocab keys by length (descending) for efficient greedy matching in encode()
        # This will now include " " and "\n"
        self._sorted_vocab_keys = sorted(self._stoi.keys(), key=len, reverse=True)

        # Create the vocabulary list for the property, ordered by token ID
        self._vocab_list = [""] * (self._current_token_id - start_token_id)
        for s, i_val in self._stoi.items():
            self._vocab_list[i_val - start_token_id] = s

    # The _validate_genome_structure, encode, decode, vocab_size, and vocabulary methods
    # remain unchanged from your version. The encode loop will now correctly handle
    # the spaces and newlines because they are part of _sorted_vocab_keys.

    def _validate_genome_structure(self, text: str) -> tuple[int, int]: # Return N, M for use
        n_val = -1
        m_val = -1
        try:
            conf_match = re.search(r"_S_CONF_(.*?)_E_CONF_", text)
            if not conf_match:
                raise ValueError("Could not find _S_CONF_..._E_CONF_ block.")

            # ---- MODIFICATION HERE ----
            conf_content = conf_match.group(1).strip() # Strip leading/trailing whitespace from the captured group
            # ---- END MODIFICATION ----

            conf_parts = conf_content.split('|')

            # Ensure we have an even number of parts for key-value pairs after splitting
            # e.g. key1|val1|key2|val2 -> 4 parts. Loop for i=0, 2.
            # The loop 'for i in range(len(conf_parts) -1)' was for key,value access using i and i+1.
            # A more robust way to parse key-value pairs:
            conf_dict = {}
            if len(conf_parts) % 2 == 0: # Expecting key|value|key|value...
                for i in range(0, len(conf_parts), 2):
                    key = conf_parts[i]
                    value = conf_parts[i+1]
                    conf_dict[key] = value
            else:
                # Fallback for the previous loop structure if keys are not strictly paired as above,
                # but this indicates a potential format issue if not all are pairs.
                # The original code implies it can find keys not necessarily at start of pairs.
                # Let's stick to the original logic of finding specific keys for now,
                # as the format is "key|value|key|value..."
                pass # Sticking to original search logic below for minimal change.


            # Original search logic for specific keys (robust to order but expects exact key match)
            for i in range(0, len(conf_parts) -1, 2): # Iterate expecting key-value pairs
                key = conf_parts[i]
                value = conf_parts[i+1]
                if key == "num_inputs":
                    n_val = int(value)
                elif key == "num_outputs":
                    m_val = int(value)

            # Fallback if keys are not found with strict pairing (e.g. if format is more flexible)
            # This part is to ensure the original intent of finding the keys is preserved if the strict pairing assumption above is wrong
            # for the given data structure. The provided example IS key|value|key|value
            if n_val == -1 or m_val == -1: # If not found by strict pairing logic
                for i in range(len(conf_parts) -1):
                    if conf_parts[i] == "num_inputs" and (i + 1) < len(conf_parts):
                        n_val = int(conf_parts[i+1])
                    elif conf_parts[i] == "num_outputs" and (i + 1) < len(conf_parts):
                        m_val = int(conf_parts[i+1])

            if n_val == -1: raise ValueError("num_inputs not found or malformed in _S_CONF_.")
            if m_val == -1: raise ValueError("num_outputs not found or malformed in _S_CONF_.")

            if not (1 <= n_val <= self._max_inputs_limit):
                raise ValueError(f"Number of inputs N={n_val} out of range [1, {self._max_inputs_limit}].")
            if not (1 <= m_val <= self._max_outputs_limit):
                raise ValueError(f"Number of outputs M={m_val} out of range [1, {self._max_outputs_limit}].")
        except Exception as e:
            # Prepend a more specific context to the re-raised exception
            raise ValueError(f"Error parsing/validating _S_CONF_ block: {e}") from e


        # --- Validation for hidden nodes (ensure m_val is correctly parsed from above) ---
        try:
            nodes_section_start_idx = text.find("_S_NODES_")
            if nodes_section_start_idx == -1:
                raise ValueError("Could not find start of _S_NODES_ block.")

            # Determine end of nodes section: start of connections or end of genome if no connections
            s_conns_idx = text.find("_S_CONNS_", nodes_section_start_idx)
            # _E_GEN_ should be the ultimate fallback
            e_gen_idx = text.find("_E_GEN_", nodes_section_start_idx + len("_S_NODES_"))

            nodes_section_end_idx = -1
            if s_conns_idx != -1:
                nodes_section_end_idx = s_conns_idx
            elif e_gen_idx != -1: # No _S_CONNS_, nodes section ends before _E_GEN_
                # The content for nodes is between _S_NODES_ and this e_gen_idx
                # but we need to be more precise if _E_NODES_ is the actual boundary
                nodes_content_up_to_egen = text[nodes_section_start_idx + len("_S_NODES_") : e_gen_idx]
                # Find the last _E_NODES_ within this segment if it exists
                last_e_nodes_in_segment = nodes_content_up_to_egen.rfind("_E_NODES_")
                if last_e_nodes_in_segment != -1:
                    nodes_section_end_idx = nodes_section_start_idx + len("_S_NODES_") + last_e_nodes_in_segment + len("_E_NODES_")
                else: # No _E_NODES_ found before _E_GEN_ (empty or malformed nodes section)
                    nodes_section_end_idx = nodes_section_start_idx + len("_S_NODES_")
            else: # Neither _S_CONNS_ nor _E_GEN_ found after _S_NODES_
                raise ValueError("Could not determine end of _S_NODES_ block (missing _S_CONNS_ or _E_GEN_).")

            nodes_content_full = text[nodes_section_start_idx + len("_S_NODES_") : nodes_section_end_idx]

            hidden_node_ids_found: set[int] = set()
            # Regex to find _NODE_id|..._E_NODES_
            # The node_id is ([^|]+?) which means any character except '|', one or more times, non-greedy.
            for match in re.finditer(r"_NODE_([^|]+?)\|.*?_E_NODES_", nodes_content_full):
                node_id_str = match.group(1)
                try:
                    node_id = int(node_id_str) # Node IDs are integers
                    if m_val == -1: # Should have been parsed correctly above.
                        raise ValueError("M (num_outputs) not determined before hidden node validation.")
                    if node_id >= m_val: # Node IDs >= M are considered hidden nodes
                        hidden_node_ids_found.add(node_id)
                except ValueError as e_node_id: # Handle if node_id_str is not a valid int
                    raise ValueError(f"Invalid node_id found: '{node_id_str}'. Error: {e_node_id}") from e_node_id

            num_hidden_nodes = len(hidden_node_ids_found)
            if num_hidden_nodes > self._max_hidden_nodes_in_genome_limit:
                raise ValueError(f"Num hidden nodes {num_hidden_nodes} exceeds limit {self._max_hidden_nodes_in_genome_limit}.")
        except Exception as e:
            raise ValueError(f"Error parsing/validating _S_NODES_ or hidden count: {e}") from e
        return n_val, m_val

    @override
    def encode(self, text: str) -> list[int]:
        self._validate_genome_structure(text)

        encoded_tokens: list[int] = []
        current_pos = 0
        text_len = len(text)

        while current_pos < text_len:
            found_match = False
            # Whitespace is now handled by being in _sorted_vocab_keys
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