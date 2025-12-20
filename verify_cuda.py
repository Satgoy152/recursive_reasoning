import torch
import sys

print("=" * 70)
print("CUDA INSTALLATION VERIFICATION")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version.split()[0]}")

print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")
    
    # Test TF32
    print(f"\nTF32 for matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 for cudnn: {torch.backends.cudnn.allow_tf32}")
    
    # Test a simple CUDA operation
    print("\nTesting CUDA operation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ CUDA operations working!")
    
    # Check mixed precision support
    print("\nMixed precision support:")
    print(f"  bf16 (recommended for A40): {torch.cuda.is_bf16_supported()}")
    print(f"  fp16: Available on all GPUs")
    
else:
    print("\n⚠ CUDA NOT AVAILABLE!")
    print("\nYou're running CPU-only PyTorch. For GPU training, you need:")
    print("1. CUDA-enabled GPU (NVIDIA)")
    print("2. CUDA toolkit installed")
    print("3. PyTorch with CUDA support:")
    print("   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 70)
