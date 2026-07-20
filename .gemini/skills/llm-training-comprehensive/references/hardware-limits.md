# Hardware Limits & VRAM Reference

## GPU VRAM Capacity Table

| GPU | VRAM | Usable VRAM | FP16 Model | QLoRA 4-bit | QLoRA 8-bit | Full FT 8B |
|-----|------|-------------|------------|-------------|-------------|------------|
| T4 (Colab/Kaggle) | 16 GB | ~15.5 GB | 8B | 8B | 8B | 3B |
| P100 (Kaggle) | 16 GB | ~15.5 GB | 8B | 8B | 8B | 3B |
| RTX 3080 | 10 GB | ~9.5 GB | 3B | 7B | 7B | 1B |
| RTX 3090 / 4090 | 24 GB | ~23 GB | 13B | 13B | 13B | 7B |
| RTX 4080 | 16 GB | ~15.5 GB | 8B | 8B | 8B | 3B |
| A10G | 24 GB | ~23 GB | 13B | 13B | 13B | 7B |
| A100-40GB | 40 GB | ~38 GB | 34B | 34B | 34B | 13B |
| A100-80GB | 80 GB | ~78 GB | 70B | 70B | 70B | 34B |
| H100-80GB | 80 GB | ~78 GB | 70B | 70B | 70B | 34B |
| H100-94GB | 94 GB | ~92 GB | 70B+ | 70B+ | 70B+ | 70B |

---

## Maximum Model Size by GPU (QLoRA 4-bit)

| GPU | Max Model (params) | Notes |
|-----|-------------------|-------|
| T4 / P100 (16GB) | 7B-8B | Tight, use batch=1, seq=2048 |
| RTX 3090/4090 (24GB) | 13B | Comfortable at seq=4096 |
| A10G (24GB) | 13B | Good for single GPU |
| A100-40GB (40GB) | 34B | Can do 34B at seq=4096 |
| A100-80GB (80GB) | 70B | Single GPU 70B QLoRA |
| H100 (80GB) | 70B+ | Fastest option |

---

## Maximum Sequence Length by GPU (8B QLoRA, batch=1)

| GPU | Max Seq Len | Batch=2 | Batch=4 |
|-----|------------|---------|---------|
| T4 (16GB) | 8192 | 4096 | 2048 |
| P100 (16GB) | 8192 | 4096 | 2048 |
| RTX 3090 (24GB) | 32768 | 16384 | 8192 |
| RTX 4090 (24GB) | 32768 | 16384 | 8192 |
| A100-40GB (40GB) | 65536 | 32768 | 16384 |
| A100-80GB (80GB) | 131072+ | 65536 | 32768 |

---

## Batch Size Recommendations (8B QLoRA, seq=4096)

| GPU | Micro Batch | Grad Accum | Effective Batch | VRAM Usage |
|-----|------------|------------|-----------------|------------|
| T4 | 1 | 16 | 16 | ~14 GB |
| T4 | 2 | 8 | 16 | ~15 GB (tight) |
| RTX 3090 | 2 | 8 | 16 | ~18 GB |
| RTX 3090 | 4 | 4 | 16 | ~21 GB |
| A100-40GB | 4 | 8 | 32 | ~32 GB |
| A100-40GB | 8 | 4 | 32 | ~36 GB |
| A100-80GB | 8 | 8 | 64 | ~60 GB |
| A100-80GB | 16 | 4 | 64 | ~70 GB |

---

## Training Time Estimates (10K samples, 3 epochs)

| Config | GPU | Time | Notes |
|--------|-----|------|-------|
| QLoRA SFT 8B | T4 | ~2-3 hours | Colab free |
| QLoRA SFT 8B | RTX 3090 | ~45 min | Local |
| QLoRA SFT 8B | A100-40GB | ~20 min | Cloud |
| QLoRA SFT 13B | A100-40GB | ~35 min | Cloud |
| QLoRA SFT 70B | A100-80GB | ~2 hours | Multi-GPU needed |
| Full FT 8B | A100-40GB | ~3 hours | Needs 40GB+ |
| Full FT 8B | A100-80GB | ~2 hours | Multi-GPU |
| DPO 8B | T4 | ~1.5 hours | Half data typically |
| DPO 8B | RTX 3090 | ~30 min | |
| GRPO 8B | A100-80GB | ~3 hours | More compute |

---

## Free Tier Limits

| Platform | GPU | Weekly Hours | Session Limit | Storage | Notes |
|----------|-----|--------------|---------------|---------|-------|
| Google Colab Free | T4 | Unlimited* | 12 hr / 90 min idle | ~80 GB (ephemeral) | *Fair use, may throttle |
| Colab Pro | T4/A100 | 100 compute units | 24 hr | ~170 GB | $10/mo |
| Colab Pro+ | A100 | 500 compute units | 24 hr | ~250 GB | $50/mo |
| Kaggle Free | T4 / P100 | 30 hrs/week | No hard limit | 20 GB (/kaggle/working) | Best free option |
| Kaggle Pro | T4 / P100 | 60 hrs/week | No hard limit | 50 GB | $10/mo |

---

## Cloud GPU Pricing (Approximate $/hour)

| Provider | RTX 3090 | RTX 4090 | A10G | A100-40GB | A100-80GB | H100 |
|----------|----------|----------|------|-----------|-----------|------|
| RunPod | $0.20-0.45 | $0.35-0.70 | $0.50-0.80 | $0.80-1.20 | $1.50-2.50 | $2.50-4.00 |
| Lambda Labs | - | $0.60 | - | $1.10 | $1.80 | $2.50 |
| Vast.ai | $0.10-0.30 | $0.25-0.60 | $0.40-0.70 | $0.70-1.00 | $1.20-2.00 | $2.00-3.50 |
| Modal | - | - | - | $1.00 | $1.80 | $2.50 |

---

## VRAM Calculation Formulas

### Model Weights
```
FP16/BF16: params * 2 bytes
INT8: params * 1 byte
INT4 (NF4): params * 0.5 bytes
INT4 (GPTQ/AWQ): params * 0.5 bytes + overhead
```

### Activations (approximate)
```
activations ≈ 2 * batch * seq_len * hidden_size * num_layers * bytes_per_param
For 8B model (32 layers, 4096 hidden): ~0.5 GB per (batch=1, seq=4096) in FP16
```

### Optimizer States
```
LoRA (r=16): ~0.1-0.5 GB (only LoRA params)
QLoRA (4-bit): ~1.5 GB (optimizer in FP32 for LoRA)
Full FT (AdamW): 4 * model_params * 2 bytes (FP32) = 8x model size
Full FT (8-bit Adam): ~2x model size
```

### Gradient Accumulation
```
Effective batch = micro_batch * grad_accum * num_gpus
VRAM independent of grad_accum (mostly)
```

---

## Quick Decision Guide

**I have a T4 (Colab/Kaggle):**
- Model: 7B-8B max
- Method: QLoRA 4-bit only
- Batch: 1-2, Grad accum 8-16
- Seq len: 2048-4096
- Time: 2-4 hours for 10K samples

**I have RTX 3090/4090 (24GB):**
- Model: 13B max
- Method: QLoRA 4-bit, maybe full FT 7B
- Batch: 2-4, Grad accum 4-8
- Seq len: 4096-8192
- Time: 30-60 min for 10K samples

**I have A100-40GB (Cloud):**
- Model: 34B max
- Method: QLoRA 4-bit, full FT 13B
- Batch: 4-8, Grad accum 4-8
- Seq len: 8192-16384
- Time: 20-40 min for 10K samples

**I have A100-80GB / H100 (Cloud):**
- Model: 70B+ max
- Method: QLoRA 4-bit, full FT 34B+
- Batch: 8-16, Grad accum 4-8
- Seq len: 16384-32768
- Time: 1-3 hours for 70B QLoRA

**I want free tier:**
1. Kaggle (30h/week, T4/P100, no idle timeout) → BEST
2. Colab Free (T4, 12h session, 90 min idle) → OK for quick experiments
3. Combine both for max free compute