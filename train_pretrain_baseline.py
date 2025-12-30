"""Pretraining script for Baseline GPT-2 - designed for multi-GPU with accelerate launch."""

import os
import time
from typing import Optional, Dict, Any

# Set NCCL timeout to 2 hours to handle long data skipping/resuming times
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"

import torch
from transformers import GPT2TokenizerFast as GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.config import ModelConfig, PretrainingConfig, WandbConfig
from src.data import get_pretrain_dataloader
from src.training import setup_scheduler
from src.utils import (
    enable_tf32,
    setup_wandb,
    load_checkpoint,
    get_latest_checkpoint,
    print_model_info,
    finish_wandb,
    save_checkpoint,
    estimate_tokens_processed,
    MetricsLogger,
    get_num_parameters,
    log_metrics,
)


def train_baseline(
    model: GPT2LMHeadModel,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    accelerator: Accelerator,
    model_config: ModelConfig,
    train_config: PretrainingConfig,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Optional[str] = None,
    start_step: int = 0,
    start_epoch: int = 0,
) -> None:
    """
    Main training loop for Baseline GPT-2.
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
        max_steps = 1_000_000

    # Print training info
    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION (BASELINE)")
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

    # Progress bar
    from tqdm import tqdm
    if accelerator.is_main_process:
        pbar = tqdm(total=max_steps, initial=start_step, desc="Training")

    while global_step < max_steps:
        epoch += 1

        for batch_idx, batch in enumerate(train_dataloader):
            # Training step
            step_start_time = time.time()
            
            x_ids, y_ids = batch
            # Concatenate x and y for standard GPT-2 training
            # x_ids: (batch, seq_len_x)
            # y_ids: (batch, seq_len_y)
            input_ids = torch.cat([x_ids, y_ids], dim=1)
            labels = input_ids.clone()
            
            # Move to device (accelerator handles this usually if using prepare, but let's be safe)
            # Actually accelerator.prepare handles dataloader, so batch is already on device?
            # The original script manually moves to device in train_step.
            # Let's check if dataloader is prepared. Yes it will be.
            # But let's be explicit if needed.
            
            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            step_time = time.time() - step_start_time

            # Update step counter (only after gradient accumulation)
            if accelerator.sync_gradients:
                global_step += 1

                # Calculate tokens/sec
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                tokens_this_step = batch_size * seq_len * accelerator.num_processes
                tokens_processed += tokens_this_step
                tokens_per_sec = tokens_this_step / step_time

                # Calculate estimated FLOPS
                n_params = get_num_parameters(model)
                # 6 * N * D
                flops_this_step = 6 * n_params * tokens_this_step
                
                metrics = {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0] if scheduler else train_config.learning_rate,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_processed": tokens_processed,
                    "step_time": step_time,
                    "estimated_flops": flops_this_step,
                }

                # Update progress bar
                if accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'lr': f"{metrics['lr']:.2e}",
                        'tok/s': f"{tokens_per_sec:.0f}",
                    })

                # Evaluation
                if eval_dataloader is not None and global_step % eval_every == 0:
                    # TODO: Implement evaluation for baseline if needed
                    pass

                # Logging
                if global_step % log_every == 0 and accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    hours = elapsed / 3600
                    tokens_billion = tokens_processed / 1e9

                    pbar.write(
                        f"[Step {global_step}/{max_steps}] "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"Tokens/sec: {tokens_per_sec:.0f} | "
                        f"Time: {hours:.1f}h | "
                        f"Tokens: {tokens_billion:.2f}B"
                    )

                    # Log to file
                    logger.log(global_step, epoch, metrics)

                    # Prepare wandb metrics
                    wandb_metrics = {
                        "train/loss": metrics['loss'],
                        "train/lr": metrics['lr'],
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_processed": tokens_processed,
                        "train/step_time": step_time,
                        "train/estimated_flops": flops_this_step,
                    }

                    # Log to WandB
                    log_metrics(wandb_metrics, global_step, accelerator)

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

    # Close progress bar
    if accelerator.is_main_process:
        pbar.close()


def main():
    # ========================================================================
    # Configuration
    # ========================================================================

    # Model configuration
    model_config = ModelConfig(
        base_model="gpt2",  # GPT-2 Small (124M)
        n_latents=0, # No latents for baseline
        n_sup=0, # No supervision for baseline
        t_loops=0, # No loops for baseline
        seq_len_x=512,
        seq_len_y=512,
        use_rope=True, # Baseline usually uses learned pos emb, but let's stick to config or standard GPT2?
                       # Standard GPT2 uses learned absolute embeddings.
                       # If we use GPT2LMHeadModel, it uses learned embeddings by default.
                       # If we want RoPE, we'd need a custom model or modify GPT2.
                       # The user said "train a normal GPT2 base model".
                       # So I will stick to standard GPT2 (learned embeddings).
        gradient_checkpointing=False,
    )

    # Pretraining configuration
    train_config = PretrainingConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_subset="sample-10BT",
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=300,
        target_tokens=2_500_000_000,
        batch_size_per_gpu=8,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        checkpoint_every=2500,
        eval_every=500,
        log_every=10,
        eval_samples=1000,
        mixed_precision="bf16",
        seed=42,
    )

    # WandB configuration
    wandb_config = WandbConfig(
        project="recursive-reasoning-trm",
        name="pretrain-gpt2-baseline-10B",
        tags=["pretraining", "gpt2-baseline", "fineweb-edu", "10B-tokens"],
        notes="Pretraining Baseline GPT-2 Small on FineWeb-Edu (10B tokens)",
        mode="online",
    )

    # Paths
    CHECKPOINT_DIR = "checkpoints/pretrain_baseline"
    LOG_DIR = "logs/pretrain_baseline"

    # Resume from checkpoint?
    RESUME = True

    # ========================================================================
    # Setup
    # ========================================================================

    # Enable TF32 for A40 speedup
    enable_tf32()

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=train_config.mixed_precision,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        log_with="wandb" if wandb_config.mode == "online" else None,
    )

    if accelerator.is_main_process:
        print(f"✓ Accelerator initialized")
        print(f"  Device: {accelerator.device}")
        print(f"  Num processes: {accelerator.num_processes}")
        print(f"  Mixed precision: {train_config.mixed_precision}")

    # Set random seed
    torch.manual_seed(train_config.seed)

    # ========================================================================
    # Model & Tokenizer
    # ========================================================================

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Initialize Baseline GPT-2 Model
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.seq_len_x + model_config.seq_len_y,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_inner=model_config.n_inner,
        activation_function="gelu_new",
        resid_pdrop=model_config.resid_pdrop,
        embd_pdrop=model_config.embd_pdrop,
        attn_pdrop=model_config.attn_pdrop,
        use_cache=False, # Gradient checkpointing usually requires this off
    )
    
    model = GPT2LMHeadModel(gpt2_config)

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print_model_info(model)

    # ========================================================================
    # Optimizer & Scheduler
    # ========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # Calculate max steps for scheduler
    if hasattr(train_config, "target_tokens"):
        batch_size = train_config.batch_size_per_gpu
        seq_len = model_config.seq_len_x + model_config.seq_len_y
        num_gpus = accelerator.num_processes
        tokens_per_step = batch_size * seq_len * num_gpus * train_config.gradient_accumulation_steps
        max_steps = train_config.target_tokens // tokens_per_step
    else:
        max_steps = 1_000_000

    scheduler = setup_scheduler(
        optimizer=optimizer,
        train_config=train_config,
        num_training_steps=max_steps,
    )

    # ========================================================================
    # Load Checkpoint (if resuming)
    # ========================================================================

    start_step = 0
    start_epoch = 0

    if RESUME:
        try:
            latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
            if latest_checkpoint:
                print(f"Found checkpoint: {latest_checkpoint}")
                checkpoint_info = load_checkpoint(
                    latest_checkpoint, 
                    model, 
                    optimizer, 
                    scheduler, 
                    accelerator=None # Not wrapped yet
                )
                start_step = checkpoint_info["step"]
                start_epoch = checkpoint_info["epoch"]
                print(f"Resuming from step {start_step}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting from scratch...")

    # ========================================================================
    # Data
    # ========================================================================

    # Calculate how many samples to skip
    samples_per_step = (
        train_config.batch_size_per_gpu
        * train_config.gradient_accumulation_steps
        * accelerator.num_processes
    )
    skip_samples = start_step * samples_per_step
    
    if skip_samples > 0:
        print(f"Skipping {skip_samples} samples...")

    # Create dataloaders
    train_dataloader = get_pretrain_dataloader(
        config=train_config,
        model_config=model_config,
        tokenizer=tokenizer,
        split="train",
        skip_samples=skip_samples,
        num_shards=accelerator.num_processes,
        shard_idx=accelerator.process_index,
    )

    # Validation dataloader (optional, skipping for now or implement if needed)
    val_dataloader = None 

    # ========================================================================
    # Prepare with Accelerator
    # ========================================================================

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Initialize WandB
    if accelerator.is_main_process:
        setup_wandb(wandb_config, model_config, train_config)

    # ========================================================================
    # Train
    # ========================================================================

    train_baseline(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        resume_from=None, # Already handled manually
        start_step=start_step,
        start_epoch=start_epoch,
    )

    if accelerator.is_main_process:
        finish_wandb()


if __name__ == "__main__":
    main()
