import argparse
import math
import os
import sys
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataloader import FileTextProvider, FolderTextProvider, TextDataset, TextProvider
from .metering import AverageValueMeter
from .model_io import (
    get_best_model_weights_path,
    get_latest_model_weights_path,
    get_tokenizer,
    load_model,
    save_tokenizer,
)

torch.manual_seed(1337)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_EPOCHS = 50
LEARNING_RATE = 3e-4
N_EMBED = 384
NUM_HEADS = 6
NUM_BLOCKS = 6


def parse_args() -> argparse.Namespace:
    """
    Create CLI args parser and execute it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_source", help="The file you want to train on", required=True, type=str)
    parser.add_argument(
        "-e",
        "--experiment",
        help="The path to the folder where to save experiment checkpoints",
        required=True,
        type=str,
    )
    return parser.parse_args()


def print_model_complexity(model: nn.Module) -> None:
    """
    Helper function to report the complexity of the model
    """
    input_shape = (BLOCK_SIZE,)

    flops, params = 0, 0  # get_model_complexity_info(model, input_shape, print_per_layer_stat=True, as_strings=True)

    print(flops)
    print(params)


class Head(nn.Module):
    def __init__(self, embedding_size: int, block_size: int, head_size: int) -> None:
        super().__init__()

        self._key = nn.Linear(embedding_size, head_size, bias=False)
        self._query = nn.Linear(embedding_size, head_size, bias=False)
        self._value = nn.Linear(embedding_size, head_size, bias=False)
        self._dropout_factor = 0.2
        self._dropout = nn.Dropout(self._dropout_factor)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, context: Tensor) -> Tensor:
        B, T, C = context.shape
        key = self._key(context)
        query = self._query(context)

        normalization_term: float = 1.0 / math.sqrt(C)
        attention_scores = query @ key.transpose(-2, -1) * normalization_term
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_scores = self._dropout(attention_scores)

        value = self._value(context)

        out: Tensor = attention_scores @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, block_size: int, head_size: int) -> None:
        super().__init__()
        self._dropout_factor = 0.2
        self._heads = nn.ModuleList(Head(embedding_size, block_size, head_size) for _ in range(num_heads))
        self._proj = nn.Linear(embedding_size, embedding_size)
        self._dropout = nn.Dropout(self._dropout_factor)

    def forward(self, context: Tensor) -> Tensor:
        out = torch.cat([head(context) for head in self._heads], dim=-1)
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
    ) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self._self_attn_heads = MultiHeadAttention(num_heads, embedding_size, block_size, head_size)
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
        num_heads: int,
        vocab_size: int,
        embedding_size: int,
        block_size: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self._block_size = block_size
        self._token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self._position_embedding_table = nn.Embedding(block_size, embedding_size)
        self._blocks = nn.Sequential(*[Block(num_heads, embedding_size, block_size) for _ in range(num_blocks)])
        self._block_norm = nn.LayerNorm(embedding_size)
        self._lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, context: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        context = context.long()
        B, T = context.shape

        tok_emb = self._token_embedding_table(context)  # B, T, C
        pos_emb = self._position_embedding_table(torch.arange(T, device=DEVICE))  # (T, C)
        x = tok_emb + pos_emb  # B, T, C
        x = self._blocks(x)
        x = self._block_norm(x)
        logits = self._lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = torch.empty(0)
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            cropped_context = context[:, -self._block_size :]
            logits, _loss = self(cropped_context)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
        return context


DatasetType = tuple[Tensor, Tensor]


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[DatasetType],
    device: torch.device,
    stage: Literal["train", "validation", "test"],
    optimizer: Optimizer | None = None,
) -> tuple[float, float]:
    """
    Run a single epoch of training, validation, or testing.

    Args:
        model: The model to run the epoch on.
        dataloader: The DataLoader to use for the epoch.
        device: The device to run on (e.g., 'cuda' or 'cpu').
        stage: The stage of the epoch ('train', 'validation', or 'test').
        optimizer: The optimizer to use for training (only used if stage is 'train').

    Returns:
        A tuple containing the mean and standard deviation of the loss for the epoch.
    """
    average_loss = AverageValueMeter()

    is_train = stage == "train"
    model.train(is_train)

    pbar = tqdm(total=len(dataloader), desc=stage.capitalize(), file=sys.stdout)

    with torch.set_grad_enabled(is_train):
        for batch, targets in dataloader:
            batch = batch.to(device)
            targets = targets.to(device)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits, loss = model(batch, targets)

            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()

            average_loss.add(loss.item())

            mean, std = average_loss.value()
            pbar.set_description(f"📉 {stage.capitalize()} Loss mean: {mean:.4f}    std: {std:.4f}")
            pbar.update(1)

    pbar.close()
    return average_loss.value()


def get_text_provider(path: str) -> TextProvider:
    if os.path.isdir(path):
        return FolderTextProvider(path)
    return FileTextProvider(path)


def main() -> None:
    args = parse_args()
    print(f"Using the device: {DEVICE}")

    text_provider = get_text_provider(args.train_source)

    tokenizer = get_tokenizer(args.experiment)

    train_dataset = TextDataset(text_provider, tokenizer, BLOCK_SIZE, "train", 0.9)
    val_dataset = TextDataset(text_provider, tokenizer, BLOCK_SIZE, "validation", 0.1)

    cpu_count = os.cpu_count() or 4
    train_dataloader = DataLoader(
        train_dataset, BATCH_SIZE, pin_memory=True, num_workers=int(cpu_count / 2), shuffle=True
    )

    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, pin_memory=True, num_workers=int(cpu_count / 2), shuffle=False)

    best_model_path = get_best_model_weights_path(args.experiment)
    latest_model_path = get_latest_model_weights_path(args.experiment)

    model = TransformerLanguageModel(NUM_HEADS, tokenizer.vocab_size, N_EMBED, BLOCK_SIZE, NUM_BLOCKS)
    load_model(best_model_path, model, DEVICE)

    print_model_complexity(model)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment, exist_ok=True)

    save_tokenizer(args.experiment, tokenizer)

    try:
        for epoch in range(MAX_EPOCHS):
            print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")

            train_loss_mean, train_loss_std = run_epoch(
                model=model, dataloader=train_dataloader, device=DEVICE, stage="train", optimizer=optimizer
            )
            print(f"Training Loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}")
            torch.save(model.state_dict(), latest_model_path)

            val_loss_mean, val_loss_std = run_epoch(
                model=model, dataloader=val_dataloader, device=DEVICE, stage="validation"
            )
            print(f"Validation Loss: {val_loss_mean:.4f} ± {val_loss_std:.4f}")

            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                print(f"Saving new best model @ {best_model_path} with validation loss: {val_loss_mean:.4f}")
                torch.save(model.state_dict(), best_model_path)

            print()
    except KeyboardInterrupt:
        torch.save(model.state_dict(), latest_model_path)
        print("Trying my best here")

    prompt = input("Tell me your prompt: ")
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(DEVICE)
    generated = model.generate(context, max_new_tokens=500)
    first_batch_trained = tokenizer.decode(generated[0].tolist())
    print(first_batch_trained)


if __name__ == "__main__":
    main()
