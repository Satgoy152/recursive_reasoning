"""Data loading and streaming utilities for TRM training."""

from typing import Iterator, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2Tokenizer

from .config import ModelConfig, PretrainingConfig, InstructionTuningConfig


# ============================================================================
# Pretaining Dataset (FineWeb-Edu)
# ============================================================================


class StreamingPretrainingDataset(IterableDataset):
    """
    Streaming dataset for pretraining on FineWeb-Edu.

    Streams documents from HuggingFace and chunks them into (x, y) pairs
    for the TRM training loop.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_subset: str,
        tokenizer: GPT2Tokenizer,
        seq_len_x: int,
        seq_len_y: int,
        split: str = "train",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer = tokenizer
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.split = split
        self.seed = seed
        self.total_seq_len = seq_len_x + seq_len_y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over (x, y) pairs."""
        # Load streaming dataset
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_subset,
            split=self.split,
            streaming=True,
        )

        # Shuffle with seed
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)

        # Buffer for accumulating tokens
        token_buffer = []

        for example in dataset:
            # Tokenize text
            text = example["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            token_buffer.extend(tokens)

            # Yield chunks when we have enough tokens
            while len(token_buffer) >= self.total_seq_len:
                chunk = token_buffer[:self.total_seq_len]
                token_buffer = token_buffer[self.total_seq_len:]

                # Split into x and y
                x_tokens = chunk[:self.seq_len_x]
                y_tokens = chunk[self.seq_len_x:]

                # Convert to tensors
                x_ids = torch.tensor(x_tokens, dtype=torch.long)
                y_ids = torch.tensor(y_tokens, dtype=torch.long)

                yield x_ids, y_ids


def get_pretrain_dataloader(
    config: PretrainingConfig,
    model_config: ModelConfig,
    tokenizer: GPT2Tokenizer,
    split: str = "train",
) -> DataLoader:
    """
    Create dataloader for pretraining.

    Args:
        config: Pretraining configuration
        model_config: Model configuration
        tokenizer: GPT2 tokenizer
        split: Dataset split ("train" or "validation")

    Returns:
        DataLoader for streaming pretraining data
    """
    dataset = StreamingPretrainingDataset(
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        tokenizer=tokenizer,
        seq_len_x=model_config.seq_len_x,
        seq_len_y=model_config.seq_len_y,
        split=split,
        seed=config.seed,
    )

    # Note: batch_size handled per-GPU by Accelerator
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size_per_gpu,
        num_workers=0,  # Reduce memory usage - no parallel data loading
        pin_memory=False,  # Disable to save memory
    )

    return dataloader


# ============================================================================
# Instruction Tuning Dataset (OpenAssistant / Alpaca)
# ============================================================================


class StreamingInstructionDataset(IterableDataset):
    """
    Streaming dataset for instruction tuning.

    Supports multiple formats:
    - OpenAssistant: "### Human: ... ### Assistant: ..."
    - Alpaca: Instruction/Input/Output format
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: GPT2Tokenizer,
        seq_len_x: int,
        seq_len_y: int,
        format: str = "openassistant",
        human_prefix: str = "### Human:",
        assistant_prefix: str = "### Assistant:",
        split: str = "train",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.format = format
        self.human_prefix = human_prefix
        self.assistant_prefix = assistant_prefix
        self.split = split
        self.seed = seed

    def _format_openassistant(self, example: dict) -> Tuple[str, str]:
        """Format OpenAssistant example into (prompt, response)."""
        # OpenAssistant format: {"text": "### Human: ... ### Assistant: ..."}
        text = example["text"]

        # Split into human and assistant parts
        parts = text.split(self.assistant_prefix)
        if len(parts) < 2:
            return None, None

        human_part = parts[0].strip()
        assistant_part = parts[1].strip()

        # Format prompt and response
        prompt = f"{human_part}\n{self.assistant_prefix}"
        response = assistant_part

        return prompt, response

    def _format_alpaca(self, example: dict) -> Tuple[str, str]:
        """Format Alpaca example into (prompt, response)."""
        # Alpaca format: {"instruction": ..., "input": ..., "output": ...}
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"{self.human_prefix} {instruction}\n{input_text}\n{self.assistant_prefix}"
        else:
            prompt = f"{self.human_prefix} {instruction}\n{self.assistant_prefix}"

        response = output

        return prompt, response

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over (x, y) pairs."""
        # Load streaming dataset
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
        )

        # Shuffle with seed
        dataset = dataset.shuffle(seed=self.seed, buffer_size=1000)

        for example in dataset:
            # Format based on dataset type
            if self.format == "openassistant":
                prompt, response = self._format_openassistant(example)
            elif self.format == "alpaca":
                prompt, response = self._format_alpaca(example)
            else:
                raise ValueError(f"Unknown format: {self.format}")

            if prompt is None or response is None:
                continue

            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

            # Ensure response ends with EOS
            if response_tokens[-1] != self.tokenizer.eos_token_id:
                response_tokens.append(self.tokenizer.eos_token_id)

            # Truncate or pad to fit sequence lengths
            if len(prompt_tokens) > self.seq_len_x:
                prompt_tokens = prompt_tokens[:self.seq_len_x]

            if len(response_tokens) > self.seq_len_y:
                response_tokens = response_tokens[:self.seq_len_y]

            # Pad if needed
            if len(prompt_tokens) < self.seq_len_x:
                padding = [self.tokenizer.pad_token_id] * (self.seq_len_x - len(prompt_tokens))
                prompt_tokens = padding + prompt_tokens  # Left padding for prompts

            if len(response_tokens) < self.seq_len_y:
                padding = [self.tokenizer.pad_token_id] * (self.seq_len_y - len(response_tokens))
                response_tokens = response_tokens + padding  # Right padding for responses

            # Convert to tensors
            x_ids = torch.tensor(prompt_tokens, dtype=torch.long)
            y_ids = torch.tensor(response_tokens, dtype=torch.long)

            yield x_ids, y_ids


def get_instruction_dataloader(
    config: InstructionTuningConfig,
    model_config: ModelConfig,
    tokenizer: GPT2Tokenizer,
    split: str = "train",
) -> DataLoader:
    """
    Create dataloader for instruction tuning.

    Args:
        config: Instruction tuning configuration
        model_config: Model configuration
        tokenizer: GPT2 tokenizer
        split: Dataset split ("train" or "test")

    Returns:
        DataLoader for streaming instruction data
    """
    dataset = StreamingInstructionDataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        seq_len_x=model_config.seq_len_x,
        seq_len_y=model_config.seq_len_y,
        format=config.dataset_format,
        human_prefix=config.human_prefix,
        assistant_prefix=config.assistant_prefix,
        split=split,
        seed=config.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size_per_gpu,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


# ============================================================================
# Validation Dataset (Fixed sample for evaluation)
# ============================================================================


class FixedValidationDataset(torch.utils.data.Dataset):
    """
    Fixed-size validation dataset for consistent evaluation.

    Pre-loads a fixed number of examples for repeatable evaluation.
    """

    def __init__(
        self,
        examples: list,
        tokenizer: GPT2Tokenizer,
        seq_len_x: int,
        seq_len_y: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x_ids, y_ids = self.examples[idx]
        return x_ids, y_ids


def create_validation_dataset(
    dataloader: DataLoader,
    num_samples: int,
) -> FixedValidationDataset:
    """
    Create a fixed validation dataset by sampling from a streaming dataloader.

    Args:
        dataloader: Streaming dataloader
        num_samples: Number of samples to collect

    Returns:
        Fixed validation dataset
    """
    examples = []
    for i, (x_ids, y_ids) in enumerate(dataloader):
        if i >= num_samples:
            break
        # Store individual examples
        for j in range(x_ids.size(0)):
            examples.append((x_ids[j], y_ids[j]))
            if len(examples) >= num_samples:
                break
        if len(examples) >= num_samples:
            break

    return examples[:num_samples]
