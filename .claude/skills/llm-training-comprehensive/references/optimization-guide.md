# Optimization Techniques Guide

## Overview

This guide covers all major optimization techniques for LLM training, ordered by impact and ease of use.

---

## 1. Flash Attention 2 (Highest Impact)

**What it does**: Replaces standard attention with memory-efficient fused kernels. Reduces attention memory from O(n²) to O(n).

**Requirements**:
- CUDA 11.8+
- GPU arch: sm_80+ (A100, RTX 3090/4090, H100)
- **NOT available on T4/P100/V100**

**Installation**:
```bash
pip install flash-attn --no-build-isolation
# Or with specific CUDA version
pip install flash-attn --no-build-isolation -f https://flashinfer.ai/whl/cu121/
```

**Usage**:
```python
# In TrainingArguments
args = TrainingArguments(
    attn_implementation="flash_attention_2",
    # OR for Unsloth (auto-detects)
    # use_gradient_checkpointing="unsloth",
)
```

**Memory savings**: 30-50% less attention memory
**Speedup**: 1.5-3x faster attention

**Verification**:
```python
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # Should be True
```

---

## 2. SDPA (Scaled Dot Product Attention)

**What it does**: PyTorch's native optimized attention. Works on more GPUs than Flash Attention 2.

**Requirements**: PyTorch 2.0+

**Usage**:
```python
args = TrainingArguments(
    attn_implementation="sdpa",
)
```

**Compatibility**: T4, P100, V100, A100, H100, RTX 30/40 series
**Memory savings**: 20-30% vs eager attention
**Speedup**: 1.2-2x

**When to use**: When Flash Attention 2 unavailable (T4, P100, V100)

---

## 3. Gradient Checkpointing

**What it does**: Trades compute for memory by recomputing activations during backward pass instead of storing them.

**Usage**:
```python
args = TrainingArguments(
    gradient_checkpointing=True,
)
```

**With Unsloth** (more efficient):
```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    ...
)
```

**Memory savings**: 30-40% less activation memory
**Speed penalty**: ~20% slower training
**When to use**: When VRAM is tight, always enable for QLoRA

---

## 4. 8-bit / 4-bit Optimizers

### 8-bit Adam (bitsandbytes)
```python
args = TrainingArguments(
    optim="adamw_bnb_8bit",
)
```
**Memory savings**: 50% less optimizer memory
**Quality**: Same as FP32 Adam
**When to use**: Always for QLoRA, recommended for LoRA

### Adafactor
```python
args = TrainingArguments(
    optim="adafactor",
)
```
**Memory savings**: No optimizer states (75% less than Adam)
**Quality**: Slightly different convergence, often needs LR adjustment
**When to use**: Full fine-tuning when VRAM very tight

### 4-bit Adam (experimental)
```python
# Via bitsandbytes 0.44+
args = TrainingArguments(
    optim="adamw_bnb_4bit",
)
```

---

## 5. GaLore (Gradient Low-Rank Projection)

**What it does**: Projects gradients to low-rank space during optimization, reducing optimizer memory by 60-80%.

**Installation**:
```bash
pip install galore-torch
```

**Usage**:
```python
from galore import GaLoreAdamW

optimizer = GaLoreAdamW(
    model.parameters(),
    lr=2e-4,
    rank=128,  # projection rank
    update_proj_gap=200,  # how often to update projection
    scale=1.0,
    proj_type="std",  # "std", "reverse_std", "left", "right", "full"
)

# Then use with Trainer
trainer = Trainer(
    model=model,
    args=args,
    optimizers=(optimizer, None),
)
```

**Memory savings**: 60-80% less optimizer memory
**Speed penalty**: ~10-15% slower
**When to use**: Full fine-tuning large models (34B+) on limited VRAM

---

## 6. Packing (Sample Packing)

**What it does**: Combines multiple short sequences into one long sequence to eliminate padding waste.

**Usage**:
```python
# Unsloth / TRL
trainer = SFTTrainer(
    packing=True,  # Enable packing
    ...
)

# Axolotl (YAML)
sample_packing: true
pad_to_sequence_len: true
```

**When to use**: 
- Dataset has many short sequences (< 1024 tokens)
- Pre-training / continued pre-training
- Single-turn instruction data

**When NOT to use**:
- Multi-turn conversations (conversations get mixed)
- When conversation structure matters
- Preference data (DPO/KTO/ORPO)

**Benefits**: 2-5x throughput improvement on short sequences

---

## 7. LoRA+ (Different Learning Rates for A/B Matrices)

**What it does**: Uses different learning rates for LoRA matrices A and B for better convergence.

**Usage** (requires PEFT 0.12+):
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", ...],
    # LoRA+ parameters
    lora_plus_scale=16,  # lr_B = lr_A * scale
    # or explicitly:
    # loq_lr=1e-5,  # Learning rate for A matrix
)
```

**Benefits**: 1-3% improvement on some benchmarks
**When to use**: When standard LoRA plateaus

---

## 8. LoRA-GA / LoftQ Initialization

**What it does**: Better initialization for LoRA using gradient information or low-rank approximation.

**Usage**:
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    init_lora_weights="loftq",  # LoRA-GA initialization
    # Options: "gaussian", "loftq", "pissa", True, False
)
```

**Benefits**: Faster convergence, better final quality
**When to use**: Always recommended (no downside)

**PiSSA** (Principal Singular Values and Singular Vectors Adaptation):
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    init_lora_weights="pissa",
)
```
Initializes LoRA with principal components of pre-trained weights.

---

## 9. LoRA Dropout Scheduling

**What it does**: Gradually reduces dropout during training for better generalization.

**Usage**:
```python
from peft import LoraConfig
from transformers import TrainerCallback

class LoRADropoutScheduler(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if hasattr(model, 'peft_config'):
            for config in model.peft_config.values():
                # Linear decay from 0.1 to 0.01 over training
                progress = state.global_step / state.max_steps
                config.lora_dropout = 0.1 * (1 - progress) + 0.01 * progress

trainer = Trainer(
    callbacks=[LoRADropoutScheduler()],
    ...
)
```

---

## 10. Gradient Accumulation

**What it does**: Accumulates gradients over multiple micro-batches before updating weights. Same effective batch size, less VRAM.

**Usage**:
```python
args = TrainingArguments(
    per_device_train_batch_size=1,  # Small micro-batch
    gradient_accumulation_steps=16,  # Effective batch = 16
)
```

**Formula**: `effective_batch = micro_batch * grad_accum * num_gpus`

**When to use**: Always when micro_batch is limited by VRAM

---

## 11. CPU Offloading (DeepSpeed ZeRO)

**What it does**: Moves optimizer states and/or parameters to CPU.

**DeepSpeed Config** (ZeRO Stage 3):
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    }
}
```

**Memory savings**: Can fit arbitrarily large models
**Speed penalty**: 3-10x slower (PCIe bandwidth limited)
**When to use**: Last resort when model doesn't fit otherwise

---

## 12. Activation Offloading (FairScale / DeepSpeed)

**What it does**: Offloads activations to CPU during forward, brings back during backward.

**Usage**:
```python
# With DeepSpeed
{
    "activation_offload": true,
    "activation_checkpointing": true
}
```

**Speed penalty**: 2-3x slower
**When to use**: When gradient checkpointing isn't enough

---

## 13. Mixed Precision

### BF16 (Recommended for Ampere+)
```python
args = TrainingArguments(
    bf16=True,  # bfloat16
)
```
- GPUs: A100, H100, RTX 3090/4090
- Better numerical stability than FP16
- Same memory as FP16

### FP16 (For older GPUs)
```python
args = TrainingArguments(
    fp16=True,
)
```
- GPUs: T4, V100, P100, RTX 20 series
- May need gradient scaling

### Auto-detect (Unsloth)
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=...,
    dtype=None,  # Auto-detects best precision
)
```

---

## 14. xFormers (Alternative to Flash Attention)

**What it does**: Memory-efficient attention with different backends.

**Installation**:
```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

**Usage**:
```python
# In model config
model.config._attn_implementation = "xformers"
```

**Compatibility**: Works on more GPUs than Flash Attention 2
**Memory savings**: Similar to Flash Attention 2

---

## 15. torch.compile (PyTorch 2.0+)

**What it does**: JIT compiles model for faster execution.

**Usage**:
```python
model = torch.compile(model, mode="max-autotune")  # Or "reduce-overhead"
```

**Speedup**: 10-30% faster training
**Caveats**: 
- First run slower (compilation)
- May have issues with dynamic shapes
- Works best with static input sizes

---

## 16. FSDP (Fully Sharded Data Parallel)

**What it does**: Shards parameters, gradients, and optimizer states across GPUs.

**Usage with Accelerate**:
```bash
accelerate launch --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer train.py
```

**Config** (YAML):
```yaml
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  backward_prefetch: backward_pre
  forward_prefetch: true
```

**When to use**: Multi-GPU, models larger than single GPU VRAM
**Alternative**: DeepSpeed ZeRO Stage 3 (similar)

---

## Optimization Priority Order

| Priority | Technique | VRAM Savings | Speed Impact | Difficulty |
|----------|-----------|--------------|--------------|------------|
| 1 | Flash Attention 2 / SDPA | 30-50% | +50-200% | Easy |
| 2 | QLoRA (4-bit) | 75% model | Minimal | Easy |
| 3 | Gradient Checkpointing | 30-40% act. | -20% | Easy |
| 4 | 8-bit Adam | 50% opt. | Minimal | Easy |
| 5 | Packing | N/A | +2-5x throughput | Easy |
| 6 | Gradient Accumulation | Enables large batch | None | Easy |
| 7 | LoRA-GA / LoftQ | N/A | Faster convergence | Easy |
| 8 | LoRA+ | N/A | +1-3% quality | Medium |
| 9 | GaLore | 60-80% opt. | -10-15% | Medium |
| 10 | FSDP / DeepSpeed | Enables multi-GPU | Varies | Hard |
| 11 | CPU Offloading | Unlimited | -3-10x | Hard |

---

## Quick Config for T4 (16GB)

```python
# Maximum VRAM savings for T4
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    fp16=True,  # T4 doesn't support bf16 well
    max_seq_length=2048,
    packing=True,
    # Flash Attention NOT available on T4
    # attn_implementation="sdpa",  # Use SDPA instead
)
```

## Quick Config for A100 (40GB)

```python
# Balanced config for A100
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    bf16=True,
    max_seq_length=4096,
    packing=True,
    attn_implementation="flash_attention_2",
)
```

## Quick Config for 70B on A100-80GB (QLoRA)

```python
# Single GPU 70B QLoRA
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    bf16=True,
    max_seq_length=4096,
    attn_implementation="flash_attention_2",
)
```

---

## Monitoring VRAM During Training

```python
import torch

def print_vram():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB")

# Call periodically in training loop or callback
```