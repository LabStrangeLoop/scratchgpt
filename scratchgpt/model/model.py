import math
from typing import Any

import torch
from ptflops import get_model_complexity_info
from torch import Tensor, nn
from torch.nn import functional as F

from scratchgpt.config import ScratchGPTConfig


class Head(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        block_size: int,
        head_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._key = nn.Linear(embedding_size, head_size, bias=False)
        self._query = nn.Linear(embedding_size, head_size, bias=False)
        self._value = nn.Linear(embedding_size, head_size, bias=False)
        self._dropout = nn.Dropout(dropout_rate)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, context: Tensor) -> Tensor:
        B, T, C = context.shape
        key = self._key(context)
        query = self._query(context)

        normalization_term: float = 1.0 / math.sqrt(C)
        attention_scores = query @ key.transpose(-2, -1) * normalization_term
        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,  # type: ignore
            float("-inf"),
        )
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_scores = self._dropout(attention_scores)

        value = self._value(context)

        out: Tensor = attention_scores @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        block_size: int,
        head_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self._heads = nn.ModuleList(Head(embedding_size, block_size, head_size, dropout_rate) for _ in range(num_heads))
        self._proj = nn.Linear(embedding_size, embedding_size)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, context: Tensor) -> Tensor:
        out: Tensor = torch.cat([head(context) for head in self._heads], dim=-1)
        out = self._proj(out)
        out = self._dropout(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self._ffwd_multipler = 4
        self._dropout = 0.2

        self._net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * self._ffwd_multipler),
            nn.ReLU(),
            nn.Linear(self._ffwd_multipler * embedding_size, embedding_size),
            nn.Dropout(self._dropout),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        out: Tensor = self._net(tensor)
        return out


class Block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        block_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self._self_attn_heads = MultiHeadAttention(
            num_heads,
            embedding_size,
            block_size,
            head_size,
            dropout_rate,
        )
        self._ffwd = FeedFoward(embedding_size)
        self._layer_norm_attention = nn.LayerNorm(embedding_size)
        self._layer_norm_ffwd = nn.LayerNorm(embedding_size)

    def forward(self, tensor: Tensor) -> Tensor:
        normal_tensor = self._layer_norm_attention(tensor)

        tensor = tensor + self._self_attn_heads(normal_tensor)

        normal_tensor = self._layer_norm_ffwd(tensor)
        tensor = tensor + self._ffwd(normal_tensor)
        return tensor


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        config: ScratchGPTConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        arch = config.architecture
        training = config.training
        assert arch.vocab_size is not None, "Must supply vocabulary size"

        self._block_size = arch.block_size
        self._token_embedding_table = nn.Embedding(arch.vocab_size, arch.embedding_size)
        self._position_embedding_table = nn.Embedding(
            arch.block_size,
            arch.embedding_size,
        )
        self._blocks = nn.Sequential(
            *[
                Block(
                    arch.num_heads,
                    arch.embedding_size,
                    arch.block_size,
                    training.dropout_rate,
                )
                for _ in range(arch.num_blocks)
            ]
        )
        self._block_norm = nn.LayerNorm(arch.embedding_size)
        self._lm_head = nn.Linear(arch.embedding_size, arch.vocab_size)
        self._device = device

    def forward(self, context: Tensor) -> Tensor:
        context = context.long()
        B, T = context.shape

        tok_emb = self._token_embedding_table(context)  # B, T, C
        pos_emb = self._position_embedding_table(torch.arange(T, device=self._device))  # (T, C)
        x = tok_emb + pos_emb  # B, T, C
        x = self._blocks(x)
        x = self._block_norm(x)
        logits: Tensor = self._lm_head(x)  # (B, T, vocab_size)
        return logits

    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            cropped_context = context[:, -self._block_size :]
            logits = self(cropped_context)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
        return context


def print_model_complexity(model: TransformerLanguageModel, config: ScratchGPTConfig, device: torch.device) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== MODEL COMPLEXITY ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    input_shape = (config.architecture.block_size,)

    def input_constructor(input_shape: Any) -> Tensor:
        return torch.randint(
            0,
            model._token_embedding_table.num_embeddings,
            (1,) + input_shape,
            device=device,
        )

    flops, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_constructor,
        print_per_layer_stat=False,
        as_strings=False,
    )

    print(f" FLOPs per forward pass: {flops:,}")
    print(f" Params: {params}")

    print("=========================")
