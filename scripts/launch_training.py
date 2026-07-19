#!/usr/bin/env python3
"""
Unified Training Launcher
Detects platform, sets up distributed training, handles checkpoint resume, and crash recovery.
"""

import os
import sys
import argparse
import yaml
import json
import subprocess
import torch
from pathlib import Path

def detect_platform():
    """Auto-detect execution platform"""
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    elif os.path.exists('/runpod'):
        return 'runpod'
    elif os.path.exists('/lambda'):
        return 'lambda'
    else:
        return 'local'

def setup_environment(platform):
    """Platform-specific setup"""
    if platform == 'colab':
        from google.colab import drive
        drive.mount('/content/drive')
        os.environ['HF_HOME'] = '/content/drive/MyDrive/llm-cache'
    elif platform == 'kaggle':
        os.environ['HF_HOME'] = '/kaggle/working/cache'
    elif platform in ['runpod', 'lambda']:
        pass  # Usually ready to go

def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_latest_checkpoint(output_dir):
    """Find latest checkpoint for resume"""
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))

def build_accelerate_command(config, platform, resume_from=None):
    """Build accelerate launch command"""
    cmd = ["accelerate", "launch"]
    
    # Multi-GPU config
    num_gpus = config.get('hardware', {}).get('num_gpus', 1)
    use_deepspeed = config.get('hardware', {}).get('use_deepspeed', False)
    use_fsdp = config.get('hardware', {}).get('use_fsdp', False)
    
    if num_gpus > 1:
        cmd.extend(["--num_processes", str(num_gpus)])
        
        if use_fsdp:
            cmd.extend([
                "--fsdp", "full_shard auto_wrap",
                "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer"
            ])
        elif use_deepspeed:
            deepspeed_config = config.get('hardware', {}).get('deepspeed_config', 'ds_config_zero3.json')
            cmd.extend(["--deepspeed", deepspeed_config])
    
    # Mixed precision
    mixed_precision = config.get('optimization', {}).get('mixed_precision', 'bf16')
    cmd.extend(["--mixed_precision", mixed_precision])
    
    # Script
    script = config.get('script', 'train.py')
    cmd.append(script)
    
    # Config file
    cmd.extend(["--config", config_path])
    
    # Resume
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", str(resume_from)])
    
    return cmd

def run_training(config_path, platform, script_path=None):
    """Main training launcher"""
    config = load_config(config_path)
    
    # Setup
    setup_environment(platform)
    
    # Find checkpoint for resume
    output_dir = config.get('output_dir', 'outputs')
    resume_from = find_latest_checkpoint(output_dir)
    
    if resume_from:
        print(f"Resuming from: {resume_from}")
    
    # Build and run command
    if script_path:
        cmd = [sys.executable, script_path, "--config", config_path]
        if resume_from:
            cmd.extend(["--resume_from_checkpoint", str(resume_from)])
    else:
        cmd = build_accelerate_command(config, platform, resume_from)
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run with crash recovery
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            if "out of memory" in str(e).lower():
                print("OOM detected, attempting recovery...")
                # Could auto-reduce batch size here
                return 1
            elif attempt < max_retries - 1:
                print(f"Training crashed (attempt {attempt + 1}/{max_retries}), retrying...")
                import time
                time.sleep(10)
            else:
                raise

def main():
    parser = argparse.ArgumentParser(description="Unified Training Launcher")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--platform", choices=['auto', 'local', 'colab', 'kaggle', 'runpod', 'lambda'], 
                       default='auto', help="Execution platform")
    parser.add_argument("--script", help="Direct Python script to run")
    args = parser.parse_args()
    
    platform = detect_platform() if args.platform == 'auto' else args.platform
    print(f"Platform: {platform}")
    print(f"GPUs: {torch.cuda.device_count()}")
    
    run_training(args.config, platform, args.script)

if __name__ == "__main__":
    main()