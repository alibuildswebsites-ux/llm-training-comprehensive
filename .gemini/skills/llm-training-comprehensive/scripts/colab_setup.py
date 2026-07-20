#!/usr/bin/env python3
"""
Colab-specific setup script - run at start of Colab notebook
"""

import os
import sys

def setup_colab():
    """Setup Google Colab environment for LLM training"""
    
    print("=" * 60)
    print("Colab LLM Training Setup")
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
    
    # 2. Mount Google Drive
    print("\n[1/5] Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        print("Drive mounted at /content/drive")
        
        # Create directories
        dirs = [
            '/content/drive/MyDrive/llm-checkpoints',
            '/content/drive/MyDrive/llm-datasets',
            '/content/drive/MyDrive/llm-outputs',
            '/content/drive/MyDrive/llm-cache',
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print("Created checkpoint/dataset/output directories")
    except Exception as e:
        print(f"Drive mount failed: {e}")
    
    # 3. Set secrets
    print("\n[2/5] Setting up secrets...")
    try:
        from google.colab import userdata
        for secret in ['HF_TOKEN', 'WANDB_API_KEY']:
            try:
                value = userdata.get(secret)
                if value:
                    os.environ[secret] = value
                    print(f"  {secret}: Set")
            except:
                print(f"  {secret}: Not found in secrets")
    except Exception as e:
        print(f"Secrets setup failed: {e}")
    
    # 4. Install dependencies
    print("\n[3/5] Installing dependencies...")
    import subprocess
    deps = [
        "unsloth",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "trl",
        "bitsandbytes",
        "wandb",
    ]
    for dep in deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], 
                         capture_output=True, check=True)
            print(f"  {dep}: OK")
        except subprocess.CalledProcessError:
            print(f"  {dep}: FAILED")
    
    # 5. Flash Attention (not on T4, skip)
    print("\n[4/5] Flash Attention: Skipping (not supported on T4)")
    
    # 6. Set cache directories
    print("\n[5/5] Setting cache directories...")
    os.environ['HF_HOME'] = '/content/drive/MyDrive/llm-cache'
    os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/llm-cache'
    os.environ['WANDB_DIR'] = '/content/drive/MyDrive/llm-cache/wandb'
    print("Cache dirs set to Google Drive")
    
    print("\n" + "=" * 60)
    print("Colab setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run keep-alive cell in separate tab")
    print("  2. Prepare dataset: python scripts/prepare_dataset.py ...")
    print("  3. Check VRAM: python scripts/estimate_vram.py ...")
    print("  4. Start training!")

# Keep-alive code for separate cell
KEEP_ALIVE_CODE = '''
import time
from IPython.display import Javascript

display(Javascript('''
    function ClickConnect() {
        console.log("Working...");
        document.querySelector("colab-toolbar-button#connect").click()
    }
    setInterval(ClickConnect, 60000)
'''))

print("Keep-alive running. Keep this cell running!")
while True:
    time.sleep(60)
'''

if __name__ == "__main__":
    setup_colab()
    print("\n--- Keep-alive code (run in SEPARATE cell) ---")
    print(KEEP_ALIVE_CODE)