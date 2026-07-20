# Troubleshooting Reference

This is the comprehensive diagnostic playbook for LLM training issues.

---

## Quick Diagnostic Commands

```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check package versions
python -c "import transformers, peft, trl, unsloth, bitsandbytes; print(transformers.__version__, peft.__version__, trl.__version__, unsloth.__version__, bitsandbytes.__version__)"

# Check VRAM usage during training
python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB')"
```

---

## 1. OOM (Out of Memory) Errors

### Error Messages
- `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X bytes.`
- `RuntimeError: CUDA out of memory`

### Root Causes
1. Model too large for VRAM
2. Batch size too large
3. Sequence length too long
4. Gradient accumulation not working
5. Memory fragmentation
6. DataLoader caching

### Solutions by Severity

#### Level 1: Config Changes (No Code Changes)
```yaml
# Reduce these in order:
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Keep effective batch ~16
max_seq_length: 2048             # Reduce from 4096/8192
```

#### Level 2: Quantization
```python
# Enable QLoRA (4-bit)
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
bnb_4bit_use_double_quant=True
```

#### Level 3: Architecture Changes
```python
# Fewer target modules
target_modules=["q_proj", "v_proj"]  # Instead of all 7

# LoRA rank
r=8  # Instead of 16/32
```

#### Level 4: Advanced
```python
# Gradient checkpointing
gradient_checkpointing=True
# Unsloth optimized:
use_gradient_checkpointing="unsloth"

# 8-bit optimizer
optim="adamw_bnb_8bit"

# CPU offloading (last resort)
# DeepSpeed ZeRO-3 with offload
```

### GPU-Specific Fixes

| GPU | Max Model (QLoRA) | Batch | Seq Len | Special |
|-----|-------------------|-------|---------|---------|
| T4 16GB | 7B-8B | 1-2 | 2048-4096 | No Flash Attn, use SDPA |
| P100 16GB | 7B-8B | 1-2 | 2048-4096 | No Flash Attn |
| RTX 3090 24GB | 13B | 2-4 | 4096-8192 | Flash Attn 2 OK |
| RTX 4090 24GB | 13B | 2-4 | 4096-8192 | Flash Attn 2 OK |
| A100 40GB | 34B | 4-8 | 8192-16384 | Flash Attn 2, BF16 |
| A100 80GB | 70B | 8-16 | 16384-32768 | Full FT 34B possible |
| H100 80GB | 70B+ | 16-32 | 32768+ | Fastest |

### Memory Fragmentation Fix
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Or in training script
import torch
torch.cuda.empty_cache()
# Periodically during training
```

---

## 2. NaN / Inf Loss

### Error Messages
- `ValueError: loss is nan, stopping training`
- `RuntimeError: Expected all values to be finite`
- `Loss became NaN at step X`

### Causes & Fixes

| Cause | Symptom | Fix |
|-------|---------|-----|
| LR too high | NaN at step 1-10 | Reduce LR 10x (2e-4 → 2e-5) |
| No gradient clipping | NaN after 100+ steps | `max_grad_norm=1.0` |
| FP16 overflow | NaN with fp16, OK with bf16 | Use `bf16=True` or lower LR |
| Bad data | NaN at specific steps | Filter dataset, check for empty/invalid |
| Pad token issue | NaN with certain sequences | `tokenizer.pad_token = tokenizer.eos_token` |
| DeepSpeed config | NaN with ZeRO-3 | Check offload settings |

### Debug Script
```python
# Add to training loop
def check_nan(loss, model, step):
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"NaN/Inf at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN grad: {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf grad: {name}")
        raise ValueError("NaN loss")
```

---

## 3. Convergence Issues

### Loss Not Decreasing

**Checklist**:
- [ ] Learning rate correct? (LoRA: 2e-4, Full: 2e-5, DPO: 5e-7)
- [ ] Data quality OK? (Inspect 50 samples)
- [ ] Data quantity sufficient? (1K+ for LoRA, 10K+ for full)
- [ ] Chat template correct? (Test with tokenizer)
- [ ] Epochs enough? (LoRA: 3, Full: 1-2, DPO: 1-3)
- [ ] Packing disabled for chat? (`packing=False`)
- [ ] Sequence length sufficient? (4096 for chat)
- [ ] LoRA rank sufficient? (Try r=32 if r=16 underfits)

### Loss Decreasing But Quality Poor

**Causes**:
- Overfitting: Reduce epochs, add dropout, more data
- Wrong template: Train template ≠ inference template
- Evaluation metric wrong: Loss ≠ quality
- Data contamination: Val set in train set

### Loss Increasing (Divergence)

**Fixes**:
- Reduce LR 10x
- Increase warmup (`warmup_ratio=0.1`)
- Add gradient clipping (`max_grad_norm=0.5`)
- Check for NaN in data

---

## 4. Chat Template Issues

### Symptoms
- Model ignores system prompt
- Generates `