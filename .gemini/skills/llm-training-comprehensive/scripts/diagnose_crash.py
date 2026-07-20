#!/usr/bin/env python3
"""
Analyze training logs for crashes (OOM, NaN, hangs)
"""

import argparse
import json
import re
import sys

def analyze_trainer_state(log_path):
    """Analyze trainer_state.json for issues"""
    with open(log_path) as f:
        state = json.load(f)
    
    print("=== Trainer State Analysis ===")
    print(f"Global step: {state.get('global_step', 'N/A')}")
    print(f"Epoch: {state.get('epoch', 'N/A')}")
    print(f"Best metric: {state.get('best_metric', 'N/A')}")
    print(f"Best step: {state.get('best_model_checkpoint', 'N/A')}")
    
    log_history = state.get('log_history', [])
    if log_history:
        print(f"\nLog entries: {len(log_history)}")
        
        # Check for NaN
        for entry in log_history:
            if 'loss' in entry:
                loss = entry['loss']
                if isinstance(loss, float) and (loss != loss or loss == float('inf') or loss == float('-inf')):
                    print(f"  WARNING: NaN/Inf loss at step {entry.get('step', '?')}: {loss}")
        
        # Check loss trend
        losses = [e.get('loss') for e in log_history if 'loss' in e and isinstance(e['loss'], (int, float))]
        if len(losses) > 10:
            recent = losses[-10:]
            early = losses[:10]
            if sum(recent) / len(recent) > sum(early) / len(early) * 1.5:
                print("  WARNING: Loss increasing significantly!")
        
        # Check learning rate
        lrs = [e.get('learning_rate') for e in log_history if 'learning_rate' in e]
        if lrs:
            print(f"  LR range: {min(lrs):.2e} - {max(lrs):.2e}")

def analyze_training_log(log_path):
    """Analyze plain text training log"""
    with open(log_path) as f:
        lines = f.readlines()
    
    print("\n=== Training Log Analysis ===")
    print(f"Total lines: {len(lines)}")
    
    # OOM detection
    oom_lines = [l for l in lines if 'CUDA out of memory' in l or 'OutOfMemoryError' in l]
    if oom_lines:
        print(f"\n!!! OOM ERRORS: {len(oom_lines)} occurrences")
        for l in oom_lines[-3:]:
            print(f"  {l.strip()}")
    
    # NaN detection
    nan_lines = [l for l in lines if 'nan' in l.lower() and ('loss' in l.lower() or 'grad' in l.lower())]
    if nan_lines:
        print(f"\n!!! NaN DETECTED: {len(nan_lines)} occurrences")
        for l in nan_lines[-3:]:
            print(f"  {l.strip()}")
    
    # Gradient explosion
    grad_lines = [l for l in lines if 'grad' in l.lower() and ('norm' in l.lower() or 'clip' in l.lower())]
    if grad_lines:
        print(f"\nGradient norm mentions: {len(grad_lines)}")
        for l in grad_lines[-3:]:
            print(f"  {l.strip()}")
    
    # Training speed
    step_times = re.findall(r'step.*?(\d+\.?\d*)\s*it/s', ' '.join(lines))
    if step_times:
        times = [float(t) for t in step_times]
        print(f"\nTraining speed: {min(times):.2f} - {max(times):.2f} it/s (avg: {sum(times)/len(times):.2f})")
    
    # Epoch progress
    epochs = re.findall(r'epoch.*?(\d+)', ' '.join(lines), re.IGNORECASE)
    if epochs:
        print(f"Epochs completed: {max(map(int, epochs))}")

def analyze_wandb_logs(log_dir):
    """Analyze wandb logs if available"""
    try:
        import wandb
        api = wandb.Api()
        # This would require run ID, skip for now
        print("\n=== W&B Analysis ===")
        print("Run `wandb sync` to upload logs for analysis")
    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Diagnose training crashes")
    parser.add_argument("--log", required=True, help="Path to trainer_state.json or training log")
    parser.add_argument("--type", choices=["auto", "trainer_state", "text"], default="auto")
    args = parser.parse_args()
    
    if args.type == "auto":
        if args.log.endswith('.json') or 'trainer_state' in args.log:
            args.type = "trainer_state"
        else:
            args.type = "text"
    
    if args.type == "trainer_state":
        analyze_trainer_state(args.log)
    else:
        analyze_training_log(args.log)
    
    # General recommendations
    print("\n=== Recommendations ===")
    print("1. OOM: Reduce batch_size, increase gradient_accumulation, enable gradient_checkpointing")
    print("2. NaN: Lower learning_rate (10x), add max_grad_norm=1.0, check data quality")
    print("3. Slow: Enable flash_attention_2, use packing, check dataloader_num_workers")
    print("4. Hang: Set dataloader_num_workers=0, increase ddp_timeout")

if __name__ == "__main__":
    main()