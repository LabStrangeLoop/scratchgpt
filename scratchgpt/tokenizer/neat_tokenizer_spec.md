# Specification: NEAT Genome Token Set

**Version:** 1.0
**Date:** May 9, 2025

## 1. Introduction

This document specifies the set of tokens used to represent serialized NEAT (NeuroEvolution of Augmenting Topologies) genomes. Each token corresponds to a distinct string component found in the canonical string representation of a genome. This token set is designed for use with a tokenizer that converts genome strings into sequences of integer IDs and vice-versa, primarily for machine learning applications.

The token set is finite and predefined based on the structural elements of the genome, configurable numerical ranges, and predefined lists of functional strings.

## 2. Token Categories

Tokens are categorized based on their role and nature within the genome string. All tokens are case-sensitive string literals.

### 2.1. Structural Markers
These tokens define the beginning and end of major sections and individual elements within the genome structure.

* `_S_GEN_`: Start of a genome definition.
* `_E_GEN_`: End of a genome definition.
* `_S_CONF_`: Start of the configuration block.
* `_E_CONF_`: End of the configuration block.
* `_S_NODES_`: Start of the nodes section.
* `_E_NODES_`: End of an individual node definition.
* `_S_CONNS_`: Start of the connections section.
* `_E_CONNS_`: End of an individual connection definition.
* `_NODE_`: Marks the beginning of an individual node's data.
* `_CONN_`: Marks the beginning of an individual connection's data.

### 2.2. Keywords
These tokens represent fixed keys used within specific sections, primarily the configuration block.

* `num_inputs`: Keyword for the number of input nodes.
* `num_outputs`: Keyword for the number of output nodes.
* `feed_forward`: Keyword for the feed-forward network property.
* `node_gene_type`: Keyword for the type of node gene.
* `connection_gene_type`: Keyword for the type of connection gene.

### 2.3. Separators and Delimiters
All tokens are separated by the following characters:

* ` ` (Space character)

### 2.4. Boolean Values
String representations of boolean values.

* `True`
* `False`

### 2.6. Gene Type Names
String identifiers for types of genes. This list is predefined below:

* *Genes:* `"DefaultNodeGene"`, `"DefaultConnectionGene"`

### 2.7. Operation Names
String identifiers for the overall operation or task the genome is designed for (e.g., specific logic gates). This list is predefined but configurable. The implementation must be able to handle accepting a new list of ops and properly assign token ids to the new operations. Example ops below:

* *Ops:* `"</xor>"`, `"</xnor>"`, `"</a_nimp_b>"`, `"</b_nimp_a>"`

### 2.8. Activation Function Names
String identifiers for neuron activation functions. This list is predefined below:

* *Activations:* `"sigmoid"`, `"relu"`, `"tanh"`, `"inv"`, `"log"`, `"abs"`, `"clamped"`

### 2.9. Aggregation Function Names
String identifiers for neuron input aggregation methods. This list is predefined below:

* *Aggregations:* `"sum"`, `"product"`, `"mean"`, `"max"`

### 2.10. Numerical Value Strings (Floats)
These tokens represent floating-point numbers (typically for biases, responses, and weights) formatted as strings.
* **Format:** Strings representing numbers with a fixed `float_precision` number of decimal places. A leading zero is expected for numbers with an absolute value less than 1 (e.g., "0.07", not ".07").
* **Range:** Generated for all representable values between a configured `float_min_val` and `float_max_val` (inclusive) at the specified precision. The implementations must be configurable to accept new values for `float_precision`, `float_min_val` and `float_max_val`
* *Example (for `float_precision=2`):* `"-2.75"`, `"-0.03"`, `"0.00"`, `"1.00"`, `"15.67"`

### 2.11. Identifier and Count Strings (Integers as Strings)
These tokens represent various integer IDs and counts, formatted as strings.

* **Input Node IDs:** Negative integers as strings.
    * *Range:* `"-1"`, `"-2"`, ..., up to `"-<max_inputs>"` (where `max_inputs` is a configurable limit, e.g., 10).
* **Output and Remapped Hidden Node IDs:** Non-negative integers as strings.
    * *Range:* `"0"`, `"1"`, ..., up to a maximum derived from `max_outputs` and `max_hidden_nodes_in_genome` (e.g., "39" if max outputs is 10 and max hidden is 30). This covers output node IDs `0` to `M-1` and renumbered hidden node IDs `M` to `M + n - 1`.
* **N (Number of Inputs) Values:** Positive integers as strings, representing the actual count of inputs for a genome.
    * *Range:* `"1"`, `"2"`, ..., up to `"<max_inputs>"`.
* **M (Number of Outputs) Values:** Positive integers as strings, representing the actual count of outputs for a genome.
    * *Range:* `"1"`, `"2"`, ..., up to `"<max_outputs>"`.

## 3. Tokenization Rules Summary

* All genome strings are a sequence of characters where all tokens are separated by either ` `. The separation characters are also tokens and must be injected back into decoded string. Its important that the original string and decoded string match exactly.
* All tokens listed above, including whitespace tokens (" "), are distinct entries in the vocabulary.
* Numerical values (floats, IDs, counts) must exactly match one of the generated string representations in the vocabulary. For floats, this means adhering to the configured `float_precision` and canonical format (e.g., "0.XX", not ".XX").
* Any sequence of characters in the input string that cannot be matched to a token in the vocabulary will result in a tokenization error.

## 4. Genome Structure for Token Context

The tokens are expected to appear in a sequence that conforms to the following general NEAT genome serialization structure:

```
S_GEN <op_name_token> S_CONF <keyword_token> <value_token> ... E_CONF S_NODES NODE <node_id_token> <bias_token> <response_token> <activation_fn_token> <aggregation_fn_token> E_NODES ... (more nodes) S_CONNS CONN <from_id_token> <to_id_token> <weight_token> E_CONNS ... (more connections) E_GEN
```