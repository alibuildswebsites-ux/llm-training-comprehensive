#!/usr/bin/env python3
"""
Kaggle-specific setup script
"""

import os
import sys

def setup_kaggle():
    """Setup Kaggle environment for LLM training"""
    
    print("=" * 60)
    print("Kaggle LLM Training Setup")
    print("=" * 60)
    
    # 1. Check GPU
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"GPU check failed: {e}")
    
    # 2. Check internet
    print("\n[1/4] Checking internet...")
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        print("Internet: OK")
    except:
        print("Internet: NOT AVAILABLE - Enable in Settings!")
    
    # 3. Setup directories
    print("\n[2/4] Setting up directories...")
    dirs = [
        '/kaggle/working/checkpoints',
        '/kaggle/working/outputs',
        '/kaggle/working/cache',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Created: /kaggle/working/{checkpoints,outputs,cache}")
    
    # 4. Set HF cache
    os.environ['HF_HOME'] = '/kaggle/working/cache'
    os.environ['TRANSFORMERS_CACHE'] = '/kaggle/working/cache'
    print(f"HF cache set to: {os.environ['HF_HOME']}")
    
    # 5. Secrets
    print("\n[3/4] Checking secrets...")
    secrets = ['HF_TOKEN', 'WANDB_API_KEY']
    for secret in secrets:
        val = os.environ.get(secret)
        if val:
            print(f"  {secret}: SET")
        else:
            print(f"  {secret}: NOT SET (add in Settings > Secrets)")
    
    # 6. Input datasets
    print("\n[4/4] Checking input datasets...")
    input_dir = '/kaggle/input'
    if os.path.exists(input_dir):
        datasets = os.listdir(input_dir)
        if datasets:
            print("Available datasets:")
            for ds in datasets:
                path = os.path.join(input_dir, ds)
                files = os.listdir(path) if os.path.isdir(path) else [ds]
                print(f"  {ds}: {files[:5]}{'...' if len(files) > 5 else ''}")
        else:
            print("No datasets mounted. Add via 'Add Data' > Your Datasets")
    else:
        print("No /kaggle/input directory found")
    
    # 7. Install deps
    print("\n[5/5] Dependencies...")
    print("Run: !pip install -q unsloth transformers datasets peft accelerate trl bitsandbytes wandb")
    
    print("\n" + "=" * 60)
    print("Setup complete! Key paths:")
    print("  Checkpoints: /kaggle/working/checkpoints/")
    print("  Outputs:     /kaggle/working/outputs/")
    print("  Cache:       /kaggle/working/cache/")
    print("  Input data:  /kaggle/input/your-dataset/")
    print("=" * 60)

if __name__ == "__main__":
    setup_kaggle()