"""Quick test script to verify installation and imports."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False

    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
        return False

    try:
        import wandb
        print(f"✓ WandB {wandb.__version__}")
    except ImportError as e:
        print(f"✗ WandB import failed: {e}")
        return False

    try:
        import einops
        print(f"✓ Einops {einops.__version__}")
    except ImportError as e:
        print(f"✗ Einops import failed: {e}")
        return False

    return True


def test_src_modules():
    """Test that all src modules can be imported."""
    print("\nTesting src modules...")

    try:
        from src.config import ModelConfig, PretrainingConfig, InstructionTuningConfig, WandbConfig
        print("✓ src.config")
    except ImportError as e:
        print(f"✗ src.config import failed: {e}")
        return False

    try:
        from src.model import HybridTRM
        print("✓ src.model")
    except ImportError as e:
        print(f"✗ src.model import failed: {e}")
        return False

    try:
        from src.data import get_pretrain_dataloader, get_instruction_dataloader
        print("✓ src.data")
    except ImportError as e:
        print(f"✗ src.data import failed: {e}")
        return False

    try:
        from src.training import train, train_step
        print("✓ src.training")
    except ImportError as e:
        print(f"✗ src.training import failed: {e}")
        return False

    try:
        from src.utils import enable_tf32, setup_wandb, save_checkpoint, load_checkpoint
        print("✓ src.utils")
    except ImportError as e:
        print(f"✗ src.utils import failed: {e}")
        return False

    return True


def test_model_creation():
    """Test that the model can be instantiated."""
    print("\nTesting model creation...")

    try:
        from src.config import ModelConfig
        from src.model import HybridTRM

        config = ModelConfig(base_model="gpt2")
        model = HybridTRM(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Latent embeddings: {config.n_latents}")
        print(f"  Base model: {config.base_model}")

        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_cuda_availability():
    """Check if CUDA is available."""
    print("\nTesting CUDA availability...")

    import torch

    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA not available (CPU mode)")
        print("  This is expected on Mac. Training requires GPU server.")

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("RECURSIVE REASONING TRM - INSTALLATION TEST")
    print("=" * 70)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_src_modules()
    all_passed &= test_model_creation()
    all_passed &= test_cuda_availability()

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Setup is complete!")
        print("\nNext steps:")
        print("1. On GPU server: Install CUDA-enabled PyTorch")
        print("2. Run notebooks/1_pretraining.ipynb for Phase 1")
        print("3. Run notebooks/2_instruction_tuning.ipynb for Phase 2")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    print("=" * 70)


if __name__ == "__main__":
    main()
