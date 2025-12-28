"""Pretraining script for HybridTRM - designed for multi-GPU with accelerate launch."""

import torch
from transformers import GPT2Tokenizer
from accelerate import Accelerator

from src.config import ModelConfig, PretrainingConfig, WandbConfig
from src.model import HybridTRM
from src.data import get_pretrain_dataloader, create_validation_dataset
from src.training import train, setup_scheduler
from src.utils import (
    enable_tf32,
    setup_wandb,
    load_checkpoint,
    get_latest_checkpoint,
    print_model_info,
    finish_wandb,
)


def main():
    # ========================================================================
    # Configuration
    # ========================================================================

    # Model configuration
    model_config = ModelConfig(
        base_model="gpt2",  # GPT-2 Small (124M)
        n_latents=64,
        n_sup=2,  # Minimum for deep supervision
        t_loops=3,
        seq_len_x=512,
        seq_len_y=512,
        use_rope=True,
        gradient_checkpointing=False,  # Enabled for memory efficiency
    )

    # Pretraining configuration
    train_config = PretrainingConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_subset="sample-10BT",
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=300,
        target_tokens=2_500_000_000,  # 1B tokens for faster iteration (change this!)
        batch_size_per_gpu=8,  # Increased from 8 - thanks to gradient checkpointing
        gradient_accumulation_steps=4,  # Adjusted for 4 GPUs (16*2*4 = 128 effective batch)
        max_grad_norm=1.0,
        checkpoint_every=2500,
        eval_every=500,
        log_every=10,  # More frequent logging to see progress
        eval_samples=1000,
        mixed_precision="bf16",  # BF16 for A40
        seed=42,
    )

    # WandB configuration
    wandb_config = WandbConfig(
        project="recursive-reasoning-trm",
        name="pretrain-gpt2-small-10B",
        tags=["pretraining", "gpt2-small", "fineweb-edu", "10B-tokens"],
        notes="Pretraining GPT-2 Small with 64 latent tokens on FineWeb-Edu (10B tokens)",
        mode="online",  # Change to "offline" if no internet
    )

    # Paths
    CHECKPOINT_DIR = "checkpoints/pretrain"
    LOG_DIR = "logs/pretrain"

    # Resume from checkpoint?
    RESUME = True

    # ========================================================================
    # Setup
    # ========================================================================

    # Enable TF32 for A40 speedup
    enable_tf32()

    # Initialize Accelerator (handles multi-GPU automatically)
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
    # Initialize Model
    # ========================================================================

    model = HybridTRM(model_config)

    if accelerator.is_main_process:
        print_model_info(model)

    # ========================================================================
    # Initialize Tokenizer
    # ========================================================================

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ========================================================================
    # Initialize Optimizer and Scheduler
    # ========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
    )

    # Calculate total training steps
    batch_size = train_config.batch_size_per_gpu
    seq_len = model_config.seq_len_x + model_config.seq_len_y
    num_gpus = accelerator.num_processes
    tokens_per_step = batch_size * seq_len * num_gpus * train_config.gradient_accumulation_steps
    num_training_steps = train_config.target_tokens // tokens_per_step

    scheduler = setup_scheduler(
        optimizer=optimizer,
        train_config=train_config,
        num_training_steps=num_training_steps,
    )

    if accelerator.is_main_process:
        print(f"✓ Optimizer and scheduler initialized")
        print(f"  Total training steps: {num_training_steps:,}")
        print(f"  Tokens per step: {tokens_per_step:,}")

    # ========================================================================
    # Load Checkpoint (if resuming)
    # ========================================================================

    start_step = 0
    start_epoch = 0

    if RESUME:
        latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint:
            if accelerator.is_main_process:
                print(f"Resuming from checkpoint: {latest_checkpoint}")
            # Note: We load into the raw model/optimizer before prepare()
            # This works because state_dicts are compatible
            metadata = load_checkpoint(
                path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=None, # Pass None because model is not yet wrapped
            )
            start_step = metadata["step"]
            start_epoch = metadata["epoch"]
        else:
            if accelerator.is_main_process:
                print("No checkpoint found, starting from scratch")
    else:
        if accelerator.is_main_process:
            print("Starting training from scratch")

    # ========================================================================
    # Initialize Dataloaders (with skip logic)
    # ========================================================================

    # Calculate how many samples to skip based on start_step
    # Each step consumes: batch_size * num_gpus * gradient_accumulation_steps samples
    # Note: This assumes the batch size hasn't changed between runs.
    
    # If each process gets its own unique stream (e.g. by sharding), then each process needs to skip:
    # start_step * batch_size_per_gpu * gradient_accumulation_steps
    
    samples_to_skip_per_process = start_step * train_config.batch_size_per_gpu * train_config.gradient_accumulation_steps

    if accelerator.is_main_process and samples_to_skip_per_process > 0:
        print(f"Skipping {samples_to_skip_per_process:,} samples to resume training...")

    # Training dataloader
    train_dataloader = get_pretrain_dataloader(
        config=train_config,
        model_config=model_config,
        tokenizer=tokenizer,
        split="train",
        skip_samples=samples_to_skip_per_process,
        num_shards=accelerator.num_processes,
        shard_idx=accelerator.process_index,
    )

    # Skip validation dataset for now to save memory
    eval_dataloader = None

    if accelerator.is_main_process:
        print(f"✓ Dataloaders initialized")

    # ========================================================================
    # Prepare for Distributed Training
    # ========================================================================

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    if accelerator.is_main_process:
        print("✓ Model and dataloaders prepared for distributed training")

    # ========================================================================
    # Initialize Weights & Biases
    # ========================================================================

    setup_wandb(
        config=wandb_config,
        train_config=train_config,
        model_config=model_config,
        accelerator=accelerator,
    )

    # ========================================================================
    # Training Loop
    # ========================================================================

    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        start_step=start_step,
        start_epoch=start_epoch,
    )

    # ========================================================================
    # Cleanup
    # ========================================================================

    finish_wandb(accelerator)

    if accelerator.is_main_process:
        print("\n✓ Pretraining completed!")
        print(f"Final checkpoint: {CHECKPOINT_DIR}/checkpoint_final.pt")


if __name__ == "__main__":
    main()
