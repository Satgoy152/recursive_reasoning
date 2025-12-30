"""Baseline Pretraining script for Standard GPT-2 - designed for multi-GPU with accelerate launch."""

import os
# Set NCCL timeout to 2 hours to handle long data skipping/resuming times
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"

import time
import math
import torch
from transformers import GPT2TokenizerFast as GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Config
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.config import ModelConfig, PretrainingConfig, WandbConfig
from src.data import get_pretrain_dataloader
from src.training import setup_scheduler
from src.utils import (
    enable_tf32,
    setup_wandb,
    save_checkpoint,
    log_metrics,
    finish_wandb,
)


def main():
    # ========================================================================
    # Configuration
    # ========================================================================

    # Model configuration - used only for sequence lengths
    model_config = ModelConfig(
        base_model="gpt2",  # GPT-2 Small (124M)
        seq_len_x=512,
        seq_len_y=512,
    )

    # Pretraining configuration - SAME as train_pretrain.py
    train_config = PretrainingConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_subset="sample-10BT",
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=300,
        target_tokens=2_500_000_000,  # 2.5B tokens
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
        name="pretrain-gpt2-baseline",
        tags=["pretraining", "gpt2-baseline", "fineweb-edu", "2.5B-tokens"],
        notes="Baseline Pretraining Standard GPT-2 Small on FineWeb-Edu (2.5B tokens)",
        mode="online",
    )

    # Paths
    CHECKPOINT_DIR = "checkpoints/pretrain_baseline"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ========================================================================
    # Setup
    # ========================================================================

    # Enable TF32
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

    # Set random seed
    torch.manual_seed(train_config.seed)

    # ========================================================================
    # Model & Tokenizer
    # ========================================================================

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Standard GPT-2 Configuration
    # We use the same hidden size (768) and layers (12) as GPT-2 Small
    # Total sequence length = seq_len_x + seq_len_y = 1024
    gpt2_config = GPT2Config(
        vocab_size=50257,
        n_positions=model_config.seq_len_x + model_config.seq_len_y,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None, # Default is 4 * n_embd
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False, # Gradient checkpointing usually requires this off or handled carefully
    )

    if train_config.gradient_checkpointing:
        gpt2_config.use_cache = False
        
    model = GPT2LMHeadModel(gpt2_config)
    
    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if accelerator.is_main_process:
        print(f"✓ Model initialized: Standard GPT-2 Small")
        print(f"  Params: {model.num_parameters() / 1e6:.1f}M")

    # ========================================================================
    # Data
    # ========================================================================

    dataloader = get_pretrain_dataloader(
        config=train_config,
        model_config=model_config,
        tokenizer=tokenizer,
        split="train",
        skip_samples=0, # No resume support needed
        num_shards=accelerator.num_processes,
        shard_idx=accelerator.process_index,
    )

    # ========================================================================
    # Optimization
    # ========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
    )

    # Calculate max steps
    total_batch_size = (
        train_config.batch_size_per_gpu
        * accelerator.num_processes
        * train_config.gradient_accumulation_steps
    )
    tokens_per_step = total_batch_size * (model_config.seq_len_x + model_config.seq_len_y)
    max_steps = int(train_config.target_tokens / tokens_per_step)

    scheduler = setup_scheduler(optimizer, train_config, max_steps)

    # Prepare with Accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Initialize WandB
    if accelerator.is_main_process:
        setup_wandb(wandb_config, {**train_config.__dict__, "model_type": "gpt2-baseline"})

    # ========================================================================
    # Training Loop
    # ========================================================================

    if accelerator.is_main_process:
        print(f"✓ Starting training for {max_steps} steps...")
        print(f"  Total batch size: {total_batch_size}")
        print(f"  Tokens per step: {tokens_per_step}")

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_main_process)
    model.train()
    
    global_step = 0
    tokens_processed = 0
    start_time = time.time()

    for step, (x_ids, y_ids) in enumerate(dataloader):
        # Concatenate x and y for standard causal LM training
        # x_ids: [batch, seq_len_x], y_ids: [batch, seq_len_y]
        input_ids = torch.cat([x_ids, y_ids], dim=1)
        
        with accelerator.accumulate(model):
            # Forward pass
            # labels=input_ids automatically shifts labels inside GPT2LMHeadModel
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
            tokens_processed += tokens_per_step

            if global_step % train_config.log_every == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed
                
                metrics = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": global_step,
                    "train/tokens_processed": tokens_processed,
                }
                
                if accelerator.is_main_process:
                    log_metrics(metrics, global_step, accelerator)

            # Checkpointing
            if global_step % train_config.checkpoint_every == 0:
                if accelerator.is_main_process:
                    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_step_{global_step}.pt"
                    print(f"Saving checkpoint to {checkpoint_path}")
                    # Simple save for baseline
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save({
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': gpt2_config,
                        'step': global_step,
                    }, checkpoint_path)

            if global_step >= max_steps:
                break

    if accelerator.is_main_process:
        print("✓ Training complete!")
        finish_wandb()

if __name__ == "__main__":
    main()
