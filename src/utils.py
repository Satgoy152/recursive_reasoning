"""Utility functions for checkpointing, logging, and monitoring."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import wandb
from accelerate import Accelerator

from .config import WandbConfig


# ============================================================================
# TF32 Acceleration for A40
# ============================================================================


def enable_tf32():
    """
    Enable TF32 (TensorFloat-32) for faster training on Ampere GPUs (A40, A100, etc.).

    TF32 provides significant speedup without loss of model quality.
    Only affects CUDA compute capability >= 8.0 (Ampere and newer).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ TF32 enabled for faster training on Ampere GPUs")


# ============================================================================
# Weights & Biases Setup
# ============================================================================


def setup_wandb(
    config: WandbConfig,
    train_config: Any,
    model_config: Any,
    accelerator: Accelerator,
) -> None:
    """
    Initialize Weights & Biases logging.

    Args:
        config: WandB configuration
        train_config: Training configuration (PretrainingConfig or InstructionTuningConfig)
        model_config: Model configuration
        accelerator: Accelerator instance
    """
    if accelerator.is_main_process:
        # Convert configs to dict
        config_dict = {
            "model": {
                "n_latents": model_config.n_latents,
                "n_sup": model_config.n_sup,
                "t_loops": model_config.t_loops,
                "seq_len_x": model_config.seq_len_x,
                "seq_len_y": model_config.seq_len_y,
                "base_model": model_config.base_model,
                "n_embd": model_config.n_embd,
                "n_layer": model_config.n_layer,
                "n_head": model_config.n_head,
                "use_rope": model_config.use_rope,
                "gradient_checkpointing": model_config.gradient_checkpointing,
            },
            "training": vars(train_config),
        }

        wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.name,
            tags=config.tags,
            notes=config.notes,
            config=config_dict,
            mode=config.mode,
        )

        print(f"✓ Weights & Biases initialized: {wandb.run.name}")


def log_metrics(metrics: Dict[str, Any], step: int, accelerator: Accelerator):
    """
    Log metrics to Weights & Biases.

    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        accelerator: Accelerator instance
    """
    if accelerator.is_main_process and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb(accelerator: Accelerator):
    """Finish Weights & Biases run."""
    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()
        print("✓ Weights & Biases run finished")


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    epoch: int,
    model_config: Any,
    train_config: Any,
    accelerator: Accelerator,
    metrics: Optional[Dict[str, Any]] = None,
):
    """
    Save training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state (optional)
        step: Current training step
        epoch: Current epoch
        model_config: Model configuration
        train_config: Training configuration
        accelerator: Accelerator instance
        metrics: Optional metrics to save with checkpoint
    """
    if accelerator.is_main_process:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Unwrap model from DDP
        unwrapped_model = accelerator.unwrap_model(model)

        # Prepare checkpoint
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "train_config": vars(train_config),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path} (step {step})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    accelerator: Optional[Accelerator] = None,
    load_optimizer: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)
        accelerator: Accelerator instance (optional)
        load_optimizer: Whether to load optimizer state

    Returns:
        Dictionary containing checkpoint metadata (step, epoch, metrics, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location="cpu")

    # Load model state
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Extract metadata
    metadata = {
        "step": checkpoint["step"],
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }

    print(f"✓ Checkpoint loaded: {path} (step {checkpoint['step']})")

    return metadata


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Find all .pt files
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))

    return str(checkpoints[-1])


# ============================================================================
# Logging Utilities
# ============================================================================


class MetricsLogger:
    """Simple logger for tracking metrics during training."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics_history = []

        # Create log directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Initialize log file
        with open(log_file, "w") as f:
            f.write("step,epoch,train_loss,lr,tokens_per_sec,tokens_processed,estimated_flops,eval_loss\n")

    def log(self, step: int, epoch: int, metrics: Dict[str, Any]):
        """Log metrics to file and memory."""
        self.metrics_history.append({"step": step, "epoch": epoch, **metrics})

        # Write to file
        with open(self.log_file, "a") as f:
            train_loss = metrics.get("loss", 0.0)
            lr = metrics.get("lr", 0.0)
            tokens_per_sec = metrics.get("tokens_per_sec", 0.0)
            tokens_processed = metrics.get("tokens_processed", 0)
            estimated_flops = metrics.get("estimated_flops", 0.0)
            eval_loss = metrics.get("eval_loss", "")
            
            f.write(f"{step},{epoch},{train_loss},{lr},{tokens_per_sec},{tokens_processed},{estimated_flops},{eval_loss}\n")

    def save_json(self, output_path: str):
        """Save metrics history as JSON."""
        with open(output_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)


# ============================================================================
# Training Utilities
# ============================================================================


def get_num_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_num_trainable_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def print_model_info(model: torch.nn.Module):
    """Print model architecture information."""
    total_params = get_num_parameters(model)
    trainable_params = get_num_trainable_parameters(model)

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print(f"Total parameters:      {format_number(total_params)} ({total_params:,})")
    print(f"Trainable parameters:  {format_number(trainable_params)} ({trainable_params:,})")
    print(f"Non-trainable:         {format_number(total_params - trainable_params)}")
    print("=" * 70 + "\n")


def estimate_tokens_processed(step: int, batch_size: int, seq_len: int, num_gpus: int) -> int:
    """
    Estimate total number of tokens processed.

    Args:
        step: Current training step
        batch_size: Batch size per GPU
        seq_len: Sequence length (seq_len_x + seq_len_y)
        num_gpus: Number of GPUs

    Returns:
        Total tokens processed
    """
    return step * batch_size * seq_len * num_gpus


def estimate_remaining_steps(
    current_step: int,
    target_tokens: int,
    batch_size: int,
    seq_len: int,
    num_gpus: int,
) -> int:
    """
    Estimate remaining steps to reach target tokens.

    Args:
        current_step: Current training step
        target_tokens: Target total tokens
        batch_size: Batch size per GPU
        seq_len: Sequence length
        num_gpus: Number of GPUs

    Returns:
        Estimated remaining steps
    """
    tokens_per_step = batch_size * seq_len * num_gpus
    total_steps = target_tokens // tokens_per_step
    return max(0, total_steps - current_step)
