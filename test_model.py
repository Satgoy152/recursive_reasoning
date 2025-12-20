"""Test script to verify the model works correctly."""

import torch
from src.config import ModelConfig
from src.model import HybridTRM

print("Testing HybridTRM model...")

# Create config
config = ModelConfig(
    base_model="gpt2",
    n_latents=64,
    n_sup=8,
    t_loops=3,
    seq_len_x=128,  # Smaller for testing
    seq_len_y=128,
    use_rope=True,
    gradient_checkpointing=False,  # Disable for testing
)

# Create model
print(f"Creating model: {config.base_model} with {config.n_latents} latent tokens...")
model = HybridTRM(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Create dummy batch
batch_size = 2
x_ids = torch.randint(0, config.vocab_size, (batch_size, config.seq_len_x))
y_ids = torch.randint(0, config.vocab_size, (batch_size, config.seq_len_y))

print(f"\nTesting forward pass...")
print(f"  Input shape: {x_ids.shape}")
print(f"  Target shape: {y_ids.shape}")

# Forward pass
try:
    model.eval()
    with torch.no_grad():
        loss, metrics = model(x_ids, y_ids)

    print(f"\n✓ Forward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    # Test generation
    print(f"\nTesting generation...")
    generated = model.generate(x_ids, max_new_tokens=10, temperature=1.0)
    print(f"  Generated shape: {generated.shape}")
    print(f"\n✓ Generation successful!")

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("✗ TEST FAILED!")
    print("=" * 70)
