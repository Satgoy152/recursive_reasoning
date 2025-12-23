"""Training loop utilities for TRM."""

import time
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_scheduler

from .model import HybridTRM
from .utils import (
    log_metrics,
    save_checkpoint,
    estimate_tokens_processed,
    MetricsLogger,
)


# ============================================================================
# Training Step
# ============================================================================


def train_step(
    model: HybridTRM,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    accelerator: Accelerator,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    step: int,
) -> Dict[str, float]:
    """
    Single training step with memory-efficient supervision loop.

    CRITICAL: We backprop INSIDE the supervision loop to avoid keeping all
    supervision losses in memory simultaneously.

    Args:
        model: HybridTRM model
        batch: Tuple of (x_ids, y_ids)
        optimizer: Optimizer
        scheduler: LR scheduler
        accelerator: Accelerator instance
        gradient_accumulation_steps: Number of accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        step: Current global step

    Returns:
        Dictionary of metrics
    """
    import torch.nn as nn

    x_ids, y_ids = batch
    x_ids = x_ids.to(accelerator.device)
    y_ids = y_ids.to(accelerator.device)

    # Get model config
    unwrapped_model = accelerator.unwrap_model(model)
    n_sup = unwrapped_model.config.n_sup
    t_loops = unwrapped_model.config.t_loops

    device = x_ids.device
    batch_size = x_ids.size(0)

    # Embed inputs
    x_embeds = unwrapped_model.transformer.wte(x_ids)

    # Initialize latent embeddings
    z_embeds = unwrapped_model.latent_embeddings.expand(batch_size, -1, -1)

    supervision_losses = []

    # CRITICAL: Deep supervision loop with IMMEDIATE backprop to save memory
    for sup_step in range(n_sup):
        # Refinement loops
        current_input = torch.cat([x_embeds, z_embeds], dim=1)

        for t in range(t_loops):
            is_last_loop = (t == t_loops - 1)

            # Only compute gradients on last loop
            with torch.set_grad_enabled(is_last_loop):
                output_embeds = unwrapped_model.forward_refine(current_input)
                z_embeds = output_embeds[:, -unwrapped_model.config.n_latents:, :]
                current_input = torch.cat([x_embeds, z_embeds], dim=1)

        # Autoregressive generation loss
        with accelerator.accumulate(model):
            logits_y = unwrapped_model.forward_ar(current_input, y_ids)

            # Compute cross-entropy loss
            shift_logits = logits_y[:, :-1, :].contiguous()
            shift_labels = y_ids[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )

            # Scale loss by n_sup (so average is correct)
            scaled_loss = loss / n_sup

            # CRITICAL: Backward pass INSIDE the loop to free memory
            accelerator.backward(scaled_loss)

            supervision_losses.append(loss.item())

        # Detach latents for next supervision step
        z_embeds = z_embeds.detach()

    # Gradient clipping and optimizer step (after all supervision steps)
    with accelerator.accumulate(model):
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    # Gather metrics
    avg_loss = sum(supervision_losses) / len(supervision_losses)
    metrics = {
        "loss": avg_loss,
        "lr": optimizer.param_groups[0]["lr"],
        "avg_supervision_loss": avg_loss,
    }

    return metrics


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(
    model: HybridTRM,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_eval_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: HybridTRM model
        dataloader: Validation dataloader
        accelerator: Accelerator instance
        max_eval_steps: Maximum number of evaluation steps (None for full dataset)

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        if max_eval_steps is not None and i >= max_eval_steps:
            break

        x_ids, y_ids = batch
        x_ids = x_ids.to(accelerator.device)
        y_ids = y_ids.to(accelerator.device)

        # Forward pass
        loss, _ = model(x_ids, y_ids)

        # Accumulate metrics
        batch_size = x_ids.size(0)
        seq_len = y_ids.size(1)
        num_tokens = batch_size * seq_len

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        num_batches += 1

    # Compute average metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_batches": num_batches,
    }


# ============================================================================
# Main Training Loop
# ============================================================================


def train(
    model: HybridTRM,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    accelerator: Accelerator,
    model_config: Any,
    train_config: Any,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Optional[str] = None,
    start_step: int = 0,
    start_epoch: int = 0,
) -> None:
    """
    Main training loop.

    Args:
        model: HybridTRM model
        train_dataloader: Training dataloader
        eval_dataloader: Validation dataloader (optional)
        optimizer: Optimizer
        scheduler: LR scheduler
        accelerator: Accelerator instance
        model_config: Model configuration
        train_config: Training configuration (PretrainingConfig or InstructionTuningConfig)
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        resume_from: Path to checkpoint to resume from
        start_step: Starting step (when resuming)
        start_epoch: Starting epoch (when resuming)
    """
    # Initialize metrics logger
    logger = MetricsLogger(f"{log_dir}/training_metrics.csv")

    # Training parameters
    gradient_accumulation_steps = train_config.gradient_accumulation_steps
    max_grad_norm = train_config.max_grad_norm
    checkpoint_every = train_config.checkpoint_every
    eval_every = train_config.eval_every
    log_every = train_config.log_every

    # Calculate total steps
    if hasattr(train_config, "max_steps") and train_config.max_steps is not None:
        max_steps = train_config.max_steps
    elif hasattr(train_config, "target_tokens"):
        # Pretraining: calculate from target tokens
        batch_size = train_config.batch_size_per_gpu
        seq_len = model_config.seq_len_x + model_config.seq_len_y
        num_gpus = accelerator.num_processes
        tokens_per_step = batch_size * seq_len * num_gpus * gradient_accumulation_steps
        max_steps = train_config.target_tokens // tokens_per_step
    else:
        # Instruction tuning: just run for a large number of steps
        max_steps = 1_000_000

    # Print training info
    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Start step:             {start_step}")
        print(f"Max steps:              {max_steps}")
        print(f"Batch size per GPU:     {train_config.batch_size_per_gpu}")
        print(f"Gradient accumulation:  {gradient_accumulation_steps}")
        print(f"Effective batch size:   {train_config.batch_size_per_gpu * gradient_accumulation_steps * accelerator.num_processes}")
        print(f"Learning rate:          {train_config.learning_rate}")
        print(f"Max grad norm:          {max_grad_norm}")
        print(f"Mixed precision:        {train_config.mixed_precision}")
        print(f"Checkpoint every:       {checkpoint_every} steps")
        print(f"Eval every:             {eval_every} steps")
        print("=" * 70 + "\n")

    # Training loop
    global_step = start_step
    epoch = start_epoch
    model.train()

    start_time = time.time()
    tokens_processed = estimate_tokens_processed(
        global_step,
        train_config.batch_size_per_gpu,
        model_config.seq_len_x + model_config.seq_len_y,
        accelerator.num_processes,
    )

    while global_step < max_steps:
        epoch += 1

        for batch_idx, batch in enumerate(train_dataloader):
            # Training step
            step_start_time = time.time()

            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                step=global_step,
            )

            step_time = time.time() - step_start_time

            # Update step counter (only after gradient accumulation)
            if accelerator.sync_gradients:
                global_step += 1

                # Calculate tokens/sec
                batch_size = batch[0].size(0)
                seq_len = model_config.seq_len_x + model_config.seq_len_y
                tokens_this_step = batch_size * seq_len * accelerator.num_processes
                tokens_processed += tokens_this_step
                tokens_per_sec = tokens_this_step / step_time

                metrics["tokens_per_sec"] = tokens_per_sec
                metrics["tokens_processed"] = tokens_processed
                metrics["step_time"] = step_time

                # Logging
                if global_step % log_every == 0 and accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    print(
                        f"[Step {global_step}/{max_steps}] "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"Tokens/sec: {tokens_per_sec:.0f} | "
                        f"Time: {elapsed:.0f}s"
                    )

                    # Log to file
                    logger.log(global_step, epoch, metrics)

                    # Log to WandB
                    log_metrics(metrics, global_step, accelerator)

                # Evaluation
                if eval_dataloader is not None and global_step % eval_every == 0:
                    if accelerator.is_main_process:
                        print(f"\n[Step {global_step}] Running evaluation...")

                    eval_metrics = evaluate(
                        model=model,
                        dataloader=eval_dataloader,
                        accelerator=accelerator,
                        max_eval_steps=100,  # Limit eval steps for speed
                    )

                    if accelerator.is_main_process:
                        print(
                            f"[Step {global_step}] "
                            f"Eval Loss: {eval_metrics['eval_loss']:.4f} | "
                            f"Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}\n"
                        )

                    # Log eval metrics
                    log_metrics(eval_metrics, global_step, accelerator)

                # Checkpointing
                if global_step % checkpoint_every == 0:
                    checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{global_step}.pt"
                    save_checkpoint(
                        path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        model_config=model_config,
                        train_config=train_config,
                        accelerator=accelerator,
                        metrics=metrics,
                    )

                # Check if we've reached max steps
                if global_step >= max_steps:
                    break

        # End of epoch
        if global_step >= max_steps:
            break

    # Final checkpoint
    if accelerator.is_main_process:
        final_checkpoint_path = f"{checkpoint_dir}/checkpoint_final.pt"
        save_checkpoint(
            path=final_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            epoch=epoch,
            model_config=model_config,
            train_config=train_config,
            accelerator=accelerator,
        )

        # Save metrics
        logger.save_json(f"{log_dir}/metrics_history.json")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Total steps:           {global_step}")
        print(f"Total tokens:          {tokens_processed:,}")
        print(f"Total time:            {total_time:.0f}s ({total_time / 3600:.2f}h)")
        print(f"Avg tokens/sec:        {tokens_processed / total_time:.0f}")
        print("=" * 70 + "\n")


# ============================================================================
# Learning Rate Scheduler Setup
# ============================================================================


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: Any,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Setup learning rate scheduler.

    Args:
        optimizer: Optimizer
        train_config: Training configuration
        num_training_steps: Total number of training steps

    Returns:
        LR scheduler
    """
    scheduler = get_scheduler(
        name=train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler
