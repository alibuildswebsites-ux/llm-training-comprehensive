#!/usr/bin/env python3
"""
Monitor training logs, detect anomalies, support early stopping
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path

def monitor_logs(log_dir, patience=5, min_delta=0.001, check_interval=30):
    """Monitor training logs for early stopping"""
    
    trainer_state_path = Path(log_dir) / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print(f"trainer_state.json not found in {log_dir}")
        return
    
    print(f"Monitoring {trainer_state_path}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"Check interval: {check_interval}s")
    print("Press Ctrl+C to stop\n")
    
    best_loss = float('inf')
    patience_counter = 0
    last_step = -1
    
    try:
        while True:
            if trainer_state_path.exists():
                with open(trainer_state_path) as f:
                    state = json.load(f)
                
                global_step = state.get('global_step', 0)
                epoch = state.get('epoch', 0)
                log_history = state.get('log_history', [])
                
                # Get latest training loss
                train_loss = None
                for entry in reversed(log_history):
                    if 'loss' in entry and 'eval_loss' not in entry:
                        train_loss = entry['loss']
                        break
                
                if train_loss is not None:
                    print(f"Step {global_step} | Epoch {epoch:.2f} | Loss: {train_loss:.4f}", end="")
                    
                    if train_loss < best_loss - min_delta:
                        best_loss = train_loss
                        patience_counter = 0
                        print("  [IMPROVED]")
                    else:
                        patience_counter += 1
                        print(f"  [no improvement: {patience_counter}/{patience}]")
                    
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered! Best loss: {best_loss:.4f}")
                        print("Create stop file to signal training to stop:")
                        print(f"  touch {log_dir}/STOP_TRAINING")
                        break
                
                last_step = global_step
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

def parse_log_file(log_path):
    """Parse training log for metrics"""
    import re
    
    with open(log_path) as f:
        content = f.read()
    
    # Extract step, loss, lr, grad_norm
    steps = re.findall(r'step.*?(\d+)', content)
    losses = re.findall(r'loss[=:]\s*([\d.]+)', content)
    lrs = re.findall(r'lr[=:]\s*([\deE\-\.]+)', content)
    grad_norms = re.findall(r'grad_norm[=:]\s*([\d.]+)', content)
    
    return {
        'steps': list(map(int, steps)),
        'losses': list(map(float, losses)),
        'lrs': list(map(float, lrs)),
        'grad_norms': list(map(float, grad_norms)),
    }

def check_anomalies(log_path):
    """Check for anomalies in training"""
    data = parse_log_file(log_path)
    
    print(f"=== Anomaly Check: {log_path} ===")
    print(f"Steps: {len(data['steps'])}, Losses: {len(data['losses'])}")
    
    if data['losses']:
        losses = data['losses']
        print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
        
        # Check for NaN
        if any(l != l for l in losses):
            print("!!! NaN LOSS DETECTED !!!")
        
        # Check for sudden increase
        if len(losses) > 5:
            recent = losses[-5:]
            prev = losses[-10:-5] if len(losses) >= 10 else losses[:-5]
            if prev and sum(recent)/len(recent) > sum(prev)/len(prev) * 2:
                print("!!! LOSS SPIKE DETECTED !!!")
        
        # Check for plateau
        if len(losses) > 100:
            recent = losses[-100:]
            if max(recent) - min(recent) < 0.001:
                print("WARNING: Loss plateaued (no change in last 100 steps)")
    
    if data['grad_norms']:
        grad_norms = data['grad_norms']
        print(f"Grad norm range: {min(grad_norms):.4f} - {max(grad_norms):.4f}")
        if any(g > 100 for g in grad_norms):
            print("!!! GRADIENT EXPLOSION DETECTED !!!")
    
    if data['lrs']:
        lrs = data['lrs']
        print(f"LR range: {min(lrs):.2e} - {max(lrs):.2e}")
        if len(set(lrs)) == 1:
            print("WARNING: LR not changing (scheduler issue?)")

def main():
    parser = argparse.ArgumentParser(description="Monitor training logs")
    parser.add_argument("--log-dir", required=True, help="Directory with trainer_state.json")
    parser.add_argument("--log-file", help="Plain text log file to analyze")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum improvement")
    parser.add_argument("--interval", type=int, default=30, help="Check interval (seconds)")
    parser.add_argument("--check-anomalies", action="store_true", help="Analyze log file for anomalies")
    args = parser.parse_args()
    
    if args.check_anomalies and args.log_file:
        check_anomalies(args.log_file)
    else:
        monitor_logs(args.log_dir, args.patience, args.min_delta, args.interval)

if __name__ == "__main__":
    main()