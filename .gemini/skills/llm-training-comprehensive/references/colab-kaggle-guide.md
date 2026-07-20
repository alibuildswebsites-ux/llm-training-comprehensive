# Colab & Kaggle Complete Guide

## Why Free Tiers Matter

- **Zero cost** for experimentation
- **Real GPUs** (T4, P100) not CPU
- **Pre-installed** CUDA, PyTorch
- **Easy sharing** via notebooks
- **Gateway** to paid cloud

---

## Google Colab Deep Dive

### Colab Free vs Pro vs Pro+

| Feature | Free | Pro ($10/mo) | Pro+ ($50/mo) |
|---------|------|--------------|---------------|
| GPU | T4 (16GB) | T4, A100-40GB | A100-40GB, A100-80GB |
| Session limit | 12 hours | 24 hours | 24 hours |
| Idle timeout | 90 min | 90 min | No timeout |
| RAM | ~12 GB | ~25 GB | ~50 GB |
| Disk | ~80 GB | ~170 GB | ~250 GB |
| Priority | Low | Medium | High |
| Background exec | No | Yes | Yes |
| Compute units | Limited | 100/mo | 500/mo |

### When to Upgrade

- **Pro**: Need A100 for 13B+ models, longer sessions
- **Pro+**: Heavy usage, background training, max reliability

### Colab Setup Checklist

```python
# 1. Verify GPU
!nvidia-smi

# 2. Mount Drive (PERSISTENCE!)
from google.colab import drive
drive.mount('/content/drive')

# 3. Set up directories
import os
os.makedirs('/content/drive/MyDrive/llm-checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/llm-datasets', exist_ok=True)
os.makedirs('/content/drive/MyDrive/llm-outputs', exist_ok=True)

# 4. Secrets (API keys)
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')

# 5. Install deps
!pip install -q unsloth transformers datasets peft accelerate trl bitsandbytes wandb

# 6. Keep-alive (run in SEPARATE cell, keep running)
import time
from IPython.display import Javascript
display(Javascript('''
    function ClickConnect() {
        console.log("Working...");
        document.querySelector("colab-toolbar-button#connect").click()
    }
    setInterval(ClickConnect, 60000)
'''))
```

### Colab Directory Structure

```
/content/                    # EPHEMERAL - lost on disconnect
├── drive/
│   └── MyDrive/
│       ├── llm-checkpoints/    # PERSISTENT - training checkpoints
│       ├── llm-datasets/       # PERSISTENT - your data
│       ├── llm-outputs/        # PERSISTENT - final models
│       └── llm-cache/          # PERSISTENT - HF cache
└── .cache/               # EPHEMERAL - pip cache, etc.
```

### Colab Training Config (T4 Optimized)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=2048,  # T4: 2048-4096
    dtype=None,
    load_in_4bit=True,  # ESSENTIAL for T4
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimized
    random_state=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,  # 2-5x speedup
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/llm-checkpoints/my-run",
        per_device_train_batch_size=1,  # T4: batch=1
        gradient_accumulation_steps=16,  # Effective batch=16
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,  # T4: fp16, not bf16
        bf16=False,
        logging_steps=10,
        optim="adamw_bnb_8bit",  # 8-bit Adam
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_steps=500,  # Frequent saves!
        save_total_limit=3,
        report_to="wandb",
        run_name="colab-t4-llama31-8b",
    ),
)

trainer.train()
```

### Colab Crash Recovery

```python
# 1. Find latest checkpoint
import glob
checkpoints = sorted(glob.glob("/content/drive/MyDrive/llm-checkpoints/my-run/checkpoint-*"))
latest = checkpoints[-1] if checkpoints else None
print(f"Resuming from: {latest}")

# 2. Resume training
trainer.train(resume_from_checkpoint=latest)

# 3. If Drive unmounted, remount and continue
```

### Colab Pro A100 Config

```python
# A100-40GB: can do 13B QLoRA, 8B full FT
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,  # Still use 4-bit for speed
)

# Larger batch possible
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch=16
    max_seq_length=4096,
    bf16=True,  # A100 supports bf16
    fp16=False,
    attn_implementation="flash_attention_2",  # A100 has FlashAttn2
)
```

---

## Kaggle Deep Dive

### Why Kaggle > Colab for Training

| Feature | Kaggle | Colab Free |
|---------|--------|------------|
| Session limit | 30h/week (no per-session limit) | 12h hard limit |
| Idle timeout | None | 90 min |
| GPU options | T4, P100 | T4 only |
| Persistence | /kaggle/working (20GB) | Drive mount required |
| Datasets | Native integration | Manual upload |
| Internet | Always on | Sometimes blocked |
| Collaboration | Native | Manual sharing |

### Kaggle Setup Checklist

```python
# 1. Verify GPU
!nvidia-smi

# 2. Enable Internet (Settings → Internet → On)

# 3. Add Dataset (your training data)
#    Add Data → Your Datasets → New Dataset → Upload JSONL/CSV
#    Path: /kaggle/input/dataset-name/

# 4. Add Secrets (Settings → Secrets)
#    HF_TOKEN, WANDB_API_KEY

# 4. Install deps
!pip install -q unsloth transformers datasets peft accelerate trl bitsandbytes wandb

# 5. Set up working directory (PERSISTS!)
import os
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/outputs', exist_ok=True)

# 6. Environment variables from secrets
import os
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
os.environ['WANDB_API_KEY'] = os.environ.get('WANDB_API_KEY', '')
```

### Kaggle Directory Structure

```
/kaggle/
├── input/
│   └── my-dataset/         # READ-ONLY - your uploaded data
│       └── train.jsonl
├── working/                # 20GB - PERSISTS across sessions!
│   ├── checkpoints/        # Training checkpoints
│   ├── outputs/            # Final models
│   └── cache/              # HF cache
└── temp/                   # Ephemeral
```

### Kaggle Training Config (T4/P100)

```python
# P100 same VRAM as T4 but older arch - similar config
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
    args=TrainingArguments(
        output_dir="/kaggle/working/checkpoints/my-run",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=10,
        optim="adamw_bnb_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb",
        run_name="kaggle-t4-llama31-8b",
        # Kaggle-specific
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        dataloader_pin_memory=False,
    ),
)

trainer.train()

# Push to Hub (best persistence)
model.push_to_hub("username/model-name", token=os.environ['HF_TOKEN'])
tokenizer.push_to_hub("username/model-name", token=os.environ['HF_TOKEN'])
```

### Kaggle P100 vs T4

```python
# P100: 16GB VRAM, Pascal arch (no FlashAttn2, no bf16)
# Same config as T4, but:
# - Slightly slower than T4 for transformers
# - No bf16 support
# - Use when T4 quota exhausted

# Check which GPU
!nvidia-smi -L
# Tesla T4 or Tesla P100-PCIE-16GB
```

### Kaggle Quota Management

```python
# Check usage (in notebook)
!curl -s "https://www.kaggle.com/api/v1/kernels/status" | python -c "
import sys, json
data = json.load(sys.stdin)
print(f'GPU hours used this week: {data.get(\"gpuHoursUsed\", 0):.1f}/30')
"

# Strategy:
# Mon-Wed: Heavy training on P100/T4
# Thu-Fri: Light experiments, eval
# Sat-Sun: Quota resets Saturday ~00:00 UTC
```

---

## Free Tier Comparison & Strategy

### Decision Matrix

| Need | Best Choice |
|------|-------------|
| Quick test (< 1hr) | Colab Free |
| Long training (8-12hr) | Kaggle (no timeout) |
| 13B+ model | Colab Pro A100 or Cloud |
| Multi-day training | Kaggle (30h/week) + checkpoints |
| Team collaboration | Kaggle |
| Persistent storage | Kaggle /kaggle/working |
| Background training | Colab Pro+ |

### Optimal Free Workflow

```
Week Start (Saturday):
1. Kaggle: Large training runs (max 30h)
2. Colab: Quick experiments, data prep, evaluation

Mid-week:
3. Kaggle: Continue training from checkpoints
4. Colab: DPO/alignment, GGUF conversion

Weekend:
5. Both: Push models to HF Hub
6. Plan next week's runs
```

### Data Transfer Between Platforms

```python
# Colab → Kaggle: Via HF Hub (recommended)
model.push_to_hub("username/model")
# In Kaggle:
model = AutoModelForCausalLM.from_pretrained("username/model")

# Colab → Kaggle: Via Google Drive
# 1. Colab: Save to /content/drive/MyDrive/model
# 2. Kaggle: Mount Drive (requires auth, slower)

# Kaggle → Colab: Via HF Hub (best)
# Or download from /kaggle/working/outputs/ via Kaggle UI
```

---

## Common Free Tier Issues & Fixes

### "CUDA out of memory"
```python
# Maximum reduction for T4
load_in_4bit=True
per_device_train_batch_size=1
gradient_accumulation_steps=32
max_seq_length=1024
gradient_checkpointing=True
target_modules=["q_proj", "v_proj"]  # Minimal LoRA
```

### "Session crashed / disconnected"
- Save checkpoints every 500 steps
- Use Drive (/content/drive/) or /kaggle/working/
- Push to HF Hub as backup
- Enable auto-reconnect (Colab)

### "Disk full"
```python
# Clear cache
!rm -rf ~/.cache/huggingface
!rm -rf /tmp/*

# Colab: Use Drive for outputs
# Kaggle: /kaggle/working has 20GB limit
```

### "Cannot install flash-attn"
- T4/P100 don't support Flash Attention 2
- Use SDPA: `attn_implementation="sdpa"`
- Or use Unsloth (handles this automatically)

### "Internet not working" (Kaggle)
- Settings → Internet → On
- Wait 30s, restart session
- Some corporate networks block Kaggle internet

### "Secret not found"
- Add in Settings → Secrets
- Restart session after adding
- Use `os.environ.get('SECRET_NAME')`

---

## Pro Tips

### 1. Use Unsloth on Free Tiers
```python
# 2-5x faster, 70% less VRAM
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Pack Your Data
```python
# 2-5x throughput on short sequences
packing=True
# But NOT for multi-turn chat!
```

### 3. Gradient Accumulation = Large Batch
```python
# Effective batch = micro_batch * grad_accum * num_gpus
# T4: 1 * 16 * 1 = 16 (good)
# Don't increase micro_batch beyond VRAM limit
```

### 4. Save to Hub, Not Local
```python
# Best persistence across platforms
model.push_to_hub("user/model", private=True)
tokenizer.push_to_hub("user/model", private=True)
```

### 5. Monitor VRAM
```python
# In training loop or callback
import torch
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

### 6. Use W&B Free Tier
```python
# Free for personal use
!wandb login
# Track: loss, lr, grad_norm, VRAM, throughput
```

---

## Checklist Before Starting Training

- [ ] GPU enabled and verified (`nvidia-smi`)
- [ ] Persistent storage configured (Drive / /kaggle/working)
- [ ] Secrets added (HF_TOKEN, WANDB_API_KEY)
- [ ] Dataset uploaded and validated
- [ ] Dependencies installed
- [ ] VRAM estimated (run estimate_vram.py)
- [ ] Checkpoint path on persistent storage
- [ ] Save frequency set (500 steps for free tier)
- [ ] W&B logging configured
- [ ] Auto-reconnect running (Colab)
- [ ] Internet enabled (Kaggle)
- [ ] Quota checked (Kaggle: 30h/week)

---

## Emergency Procedures

### Colab: 5 min left, training not done
```python
# 1. Save immediately
trainer.save_model("/content/drive/MyDrive/emergency-save")
trainer.state.save_to_json("/content/drive/MyDrive/emergency-state.json")

# 2. Push to Hub
model.push_to_hub("user/emergency-save", token=os.environ['HF_TOKEN'])

# 3. Note step number for resume
print(f"Step: {trainer.state.global_step}")
```

### Kaggle: Quota exhausted mid-week
```python
# 1. Push current model to Hub
# 2. Switch to CPU-only inference/eval
# 3. Wait for Saturday reset
# 4. Resume from Hub
```

### Both: OOM on first step
```python
# Emergency config
load_in_4bit=True
per_device_train_batch_size=1
gradient_accumulation_steps=32
max_seq_length=512
target_modules=["q_proj", "v_proj"]  # Absolute minimum
```

---

## Resources

- [Colab FAQ](https://research.google.com/colab/faq.html)
- [Kaggle GPU Quota](https://www.kaggle.com/docs/gpu)
- [Unsloth Colab Notebooks](https://github.com/unslothai/unsloth-notebooks)
- [HF Hub Docs](https://huggingface.co/docs/hub)
- [W&B Free Tier](https://wandb.ai/site/pricing)