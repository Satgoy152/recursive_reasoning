"""Configuration classes for TRM training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for HybridTRM model architecture."""

    # Latent reasoning parameters
    n_latents: int = 64
    n_sup: int = 8  # Deep supervision steps
    t_loops: int = 3  # Refinement loops per supervision step

    # Sequence lengths
    seq_len_x: int = 512  # Input sequence length
    seq_len_y: int = 512  # Target sequence length

    # Base GPT-2 architecture
    base_model: str = "gpt2"  # "gpt2" (124M) or "gpt2-medium" (355M)
    vocab_size: int = 50257
    n_embd: int = 768  # Will be auto-set based on base_model
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None  # FFN hidden size (4 * n_embd if None)

    # Dropout
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    # RoPE parameters
    use_rope: bool = True
    rope_theta: float = 10000.0

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    def __post_init__(self):
        """Auto-configure based on base_model."""
        if self.base_model == "gpt2":
            self.n_embd = 768
            self.n_layer = 12
            self.n_head = 12
        elif self.base_model == "gpt2-medium":
            self.n_embd = 1024
            self.n_layer = 24
            self.n_head = 16
        elif self.base_model == "gpt2-large":
            self.n_embd = 1280
            self.n_layer = 36
            self.n_head = 20
        elif self.base_model == "gpt2-xl":
            self.n_embd = 1600
            self.n_layer = 48
            self.n_head = 25

        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd


@dataclass
class PretrainingConfig:
    """Configuration for pretraining phase."""

    # Dataset
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    min_lr_ratio: float = 0.1  # Final LR = min_lr_ratio * learning_rate

    # Training duration
    target_tokens: int = 10_000_000_000  # 10B tokens
    max_steps: Optional[int] = None  # Auto-calculated if None

    # Batch size (total = batch_size_per_gpu * num_gpus * gradient_accumulation_steps)
    batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 1

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing and logging
    checkpoint_every: int = 5000
    eval_every: int = 2000
    log_every: int = 100

    # Validation
    eval_samples: int = 1000  # Number of samples for validation

    # Mixed precision
    mixed_precision: str = "bf16"  # "bf16", "fp16", or "no"

    # Seed
    seed: int = 42


@dataclass
class InstructionTuningConfig:
    """Configuration for instruction finetuning phase."""

    # Dataset
    dataset_name: str = "timdettmers/openassistant-guanaco"
    dataset_format: str = "openassistant"  # "openassistant" or "alpaca"

    # Training hyperparameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1

    # Training duration
    num_epochs: int = 3
    max_steps: Optional[int] = None  # Override epochs if set

    # Batch size
    batch_size_per_gpu: int = 16  # Smaller for longer sequences
    gradient_accumulation_steps: int = 2  # Effective batch size = 32 per GPU

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing and logging
    checkpoint_every: int = 500
    eval_every: int = 200
    log_every: int = 50

    # Validation
    eval_samples: int = 500

    # Mixed precision
    mixed_precision: str = "bf16"

    # Seed
    seed: int = 42

    # Special tokens for formatting
    human_prefix: str = "### Human:"
    assistant_prefix: str = "### Assistant:"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    project: str = "recursive-reasoning-trm"
    entity: Optional[str] = None  # Your wandb username/team
    name: Optional[str] = None  # Run name (auto-generated if None)
    tags: list = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = "online"  # "online", "offline", or "disabled"
