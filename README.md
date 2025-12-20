# Recursive Reasoning Transformer with Memory (TRM)

A PyTorch implementation of a hybrid transformer architecture that combines GPT-2 with learnable latent tokens and recursive refinement loops for enhanced reasoning capabilities.

## Architecture Overview

The **HybridTRM** model extends GPT-2 with:

- **Latent Memory Tokens** (64 learnable tokens): Act as a "scratchpad" for reasoning
- **Recursive Refinement Loops** (T=3): Iteratively refine latent representations
- **Deep Supervision** (N=8 steps): Multiple training signals throughout the forward pass
- **RoPE (Rotary Position Embeddings)**: Replace learned positional embeddings for better sequence handling
- **Gradient Checkpointing**: Memory-efficient training

## Project Structure

```
recursive_reasoning/
├── src/
│   ├── config.py          # Configuration classes
│   ├── model.py           # HybridTRM architecture with RoPE
│   ├── data.py            # Streaming dataloaders
│   ├── training.py        # Training loop utilities
│   └── utils.py           # Checkpointing, logging, monitoring
├── notebooks/
│   ├── 1_pretraining.ipynb       # Phase 1: Pretraining
│   └── 2_instruction_tuning.ipynb # Phase 2: Instruction finetuning
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── README.md
```

## Hardware Requirements

- **GPUs**: 2x NVIDIA A40 (48GB each) or equivalent
- **CUDA**: 11.8+ with TF32 support (Ampere architecture or newer)
- **RAM**: 64GB+ recommended for data preprocessing
- **Storage**: Minimal (datasets are streamed from HuggingFace)

## Installation

### On Your GPU Server (with CUDA)

```bash
# Clone repository
cd /path/to/your/project

# Initialize UV environment (already done)
# Install PyTorch with CUDA support
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
uv add transformers datasets accelerate wandb einops jupyter ipykernel

# Register Jupyter kernel
uv run python -m ipykernel install --user --name=recursive_reasoning
```

### On Mac (Development - CPU Only)

Packages are already installed for local development. Note: Training must be done on GPU server.

## Training Pipeline

### Phase 1: Pretraining (1-2 Days)

**Goal**: Teach the model to use latent tokens for text coherence and basic reasoning.

**Dataset**: FineWeb-Edu (sample-10BT) - ~10 billion tokens of educational web content

**Configuration**:
- Model: GPT-2 Small (124M parameters) + 64 latent tokens
- Batch size: 32 per GPU (64 total)
- Learning rate: 3e-4
- Target: 10B tokens (~152k steps)
- Checkpoints: Every 5000 steps
- Evaluation: Every 2000 steps

**Run**:
```bash
# On GPU server
cd notebooks
jupyter notebook 1_pretraining.ipynb
# Follow the notebook cells
```

**Expected Output**:
- Checkpoints in `checkpoints/pretrain/`
- Training logs in `logs/pretrain/`
- WandB dashboard with loss curves, perplexity, tokens/sec

### Phase 2: Instruction Finetuning (4-8 Hours)

**Goal**: Teach the model to follow instructions and solve problems.

**Dataset**: OpenAssistant-Guanaco (~10k curated conversations)

**Configuration**:
- Model: Load from Phase 1 checkpoint
- Batch size: 16 per GPU (32 effective with grad accumulation)
- Learning rate: 1e-5 (lower for finetuning)
- Epochs: 3
- Checkpoints: Every 500 steps
- Evaluation: Every 200 steps

**Run**:
```bash
cd notebooks
jupyter notebook 2_instruction_tuning.ipynb
# Follow the notebook cells
```

**Expected Output**:
- Finetuned checkpoints in `checkpoints/finetune/`
- Training logs in `logs/finetune/`
- Interactive generation examples

## Configuration Reference

### Model Configuration (`src/config.py`)

```python
ModelConfig(
    base_model="gpt2",           # "gpt2" (124M) or "gpt2-medium" (355M)
    n_latents=64,                # Number of latent tokens
    n_sup=8,                     # Deep supervision steps
    t_loops=3,                   # Refinement loops per supervision
    seq_len_x=512,               # Input sequence length
    seq_len_y=512,               # Target sequence length
    use_rope=True,               # Use RoPE instead of learned positions
    gradient_checkpointing=True, # Enable for memory efficiency
)
```

### Training Configuration

**Pretraining** (`PretrainingConfig`):
- Dataset: `HuggingFaceFW/fineweb-edu`
- LR: 3e-4
- Warmup: 2000 steps
- Target: 10B tokens

**Instruction Tuning** (`InstructionTuningConfig`):
- Dataset: `timdettmers/openassistant-guanaco`
- LR: 1e-5
- Warmup: 100 steps
- Epochs: 3

## Key Features

### 1. Streaming Datasets
All data is streamed from HuggingFace - no local storage needed:
```python
from src.data import get_pretrain_dataloader

dataloader = get_pretrain_dataloader(config, model_config, tokenizer)
# Automatically streams FineWeb-Edu
```

### 2. Automatic Checkpointing
```python
# Checkpoints saved every N steps with full state
checkpoint = {
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": config,
}
```

### 3. Resume Training
```python
# In notebooks, set:
RESUME = True

# Automatically loads latest checkpoint and continues training
```

### 4. Weights & Biases Integration
```python
wandb_config = WandbConfig(
    project="recursive-reasoning-trm",
    name="pretrain-gpt2-small-10B",
    mode="online",  # or "offline"
)
```

### 5. Multi-GPU Training (DDP)
Handled automatically by `Accelerator`:
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

## Performance Optimizations

### TF32 Acceleration (A40)
```python
from src.utils import enable_tf32
enable_tf32()  # ~2x speedup on Ampere GPUs
```

### Mixed Precision (bf16)
- Faster training (1.5-2x)
- Lower memory usage
- Better numerical stability than fp16

### Gradient Checkpointing
- Saves ~40% memory
- Adds ~15% compute overhead
- Essential for larger models

## Monitoring Training

### 1. Weights & Biases (Recommended)
- Loss curves
- Learning rate schedule
- Tokens/second throughput
- Evaluation perplexity
- GPU utilization

### 2. Local Logs
```bash
# CSV logs
cat logs/pretrain/training_metrics.csv

# JSON history
cat logs/pretrain/metrics_history.json
```

### 3. Console Output
```
[Step 1000/152000] Loss: 3.2456 | LR: 2.95e-04 | Tokens/sec: 45000 | Time: 180s
[Step 2000/152000] Running evaluation...
[Step 2000/152000] Eval Loss: 3.1234 | Eval Perplexity: 22.71
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size_per_gpu`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable gradient checkpointing (already enabled by default)
4. Use `gpt2` instead of `gpt2-medium`

### Slow Training
1. Ensure TF32 is enabled (`enable_tf32()`)
2. Check mixed precision is `bf16`
3. Increase `num_workers` in dataloaders
4. Verify GPUs are not throttling (check `nvidia-smi`)

### NaN Loss
1. Lower learning rate
2. Increase warmup steps
3. Check for bad data (use `eval_every` to catch early)
4. Use `bf16` instead of `fp16`

### Checkpoint Loading Errors
```python
# Load without optimizer (for finetuning)
load_checkpoint(path, model, optimizer=None, load_optimizer=False)
```

## Extending the Model

### Switching to GPT-2 Medium (355M)
```python
model_config = ModelConfig(
    base_model="gpt2-medium",  # Auto-configures n_embd, n_layer, n_head
    # ... other params
)
```

### Using Alpaca Dataset
```python
train_config = InstructionTuningConfig(
    dataset_name="your-alpaca-dataset",
    dataset_format="alpaca",  # Instead of "openassistant"
)
```

### Adjusting Latent Tokens
```python
model_config = ModelConfig(
    n_latents=128,  # More reasoning capacity
    n_sup=10,       # More supervision steps
    t_loops=5,      # More refinement iterations
)
```

## Citation

If you use this implementation, please cite:

```bibtex
@misc{recursive-reasoning-trm,
  title={Recursive Reasoning Transformer with Memory},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/recursive_reasoning}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **FineWeb-Edu**: HuggingFace for the high-quality pretraining dataset
- **OpenAssistant-Guanaco**: Tim Dettmers for the instruction dataset
- **RoFormer**: Su et al. for Rotary Position Embeddings
- **GPT-2**: OpenAI for the base architecture
- **Accelerate**: HuggingFace for multi-GPU training utilities

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
