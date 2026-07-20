---
name: llm-training-comprehensive
description: All-in-one skill for training and fine-tuning LLMs (Llama, Qwen, Mistral, Gemma, Phi) from scratch or via PEFT on Kaggle, Colab, local, or cloud GPUs.
tags: [llm, fine-tuning, lora, qlora, sft, dpo, grpo, peft, unsloth, axolotl, kaggle, colab, gguf, quantization, training]
---

# LLM Training and Fine-Tuning: Comprehensive Guide & Playbook

## Overview

This skill enables you to train, fine-tune, and export any open-source LLM on any hardware — from a free Kaggle T4 to multi-GPU cloud clusters. It covers model families (Llama, Qwen, Mistral, Gemma, Phi), training methods (SFT, DPO, KTO, ORPO, PPO, GRPO), PEFT techniques (LoRA, QLoRA, DoRA, VeRA), frameworks (Unsloth, Axolotl, LLaMA-Factory, TRL, TorchTune), and export formats (GGUF, AWQ, GPTQ).

**When to use this skill:**
- User wants to fine-tune or train an LLM
- User needs help choosing a framework, method, or hyperparameters
- User has VRAM constraints and needs optimization guidance
- User wants to export a trained model to GGUF/AWQ/GPTQ for deployment
- User needs dataset preparation or format conversion
- User is on Colab/Kaggle and needs session management strategies
- User encounters training errors (OOM, NaN, convergence issues)

---

## Quick Start

Three steps from request to training run:

### Step 1: Specify

Identify the user's requirements. Collect:
- **Model**: Which model family and size? (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- **Dataset**: What data? What format? (ShareGPT, Alpaca, OpenAI, custom)
- **Method**: SFT, DPO, KTO, ORPO, GRPO, reward modeling?
- **Hardware**: GPU type, count, platform (Colab/Kaggle/local/cloud)
- **Goal**: What should the model learn? (domain adaptation, instruction following, alignment)

### Step 2: Plan

Use the config files and scripts to build a training plan:
1. Check `config/hardware-profiles.yaml` for GPU limits
2. Run `scripts/estimate_vram.py` to validate VRAM feasibility
3. Select framework from the comparison table below
4. Choose PEFT method based on VRAM and model size
5. Set hyperparameters per the training method reference

```bash
# Example: Check if Llama-3.1-8B + QLoRA fits on T4
python scripts/estimate_vram.py --params-b 8 --seq-len 4096 --batch-size 2 --quantize 4bit
```

### Step 3: Execute

1. Prepare dataset with `scripts/prepare_dataset.py`
2. Generate framework-specific config from templates
3. Launch training
4. Monitor via W&B/TensorBoard
5. Export with merge + quantization pipeline

---

## Framework Selection Guide

### Comparison Table

| Framework | Speed | VRAM Efficiency | Multi-GPU | Config-Driven | Best For |
|-----------|-------|----------------|-----------|---------------|----------|
| **Unsloth** | 2-5x faster | 70% less VRAM | DDP/FSDP (Pro) | Python scripts | Single-GPU, free tiers |
| **Axolotl** | Fast | Good | FSDP/DeepSpeed native | YAML only | Multi-GPU, reproducibility |
| **LLaMA-Factory** | Fast | Good | DeepSpeed | YAML + WebUI | Beginners, quick experiments |
| **TRL** | Baseline | Baseline | FSDP/Accelerate | Python scripts | Custom RL, research |
| **TorchTune** | Baseline | Baseline | FSDP | YAML | PyTorch purity |
| **Raw Transformers** | Baseline | Baseline | Manual | Python scripts | Maximum control |

### Framework Details

#### Unsloth (Primary Recommendation)

Best for: Free tiers (Colab/Kaggle), single-GPU training, fastest iteration.

```bash
pip install unsloth
# Or for latest
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Key advantages:
- 2-5x faster training, 70% less VRAM via custom Triton kernels
- Drop-in replacement for HuggingFace classes
- Supports SFT, DPO, ORPO, GRPO, continued pretraining
- Automatic gradient checkpointing, Flash Attention integration
- Free tier friendly — fits 7B QLoRA on T4 with 4096 context

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    dtype=None,  # auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="outputs",
    ),
)
trainer.train()
model.save_pretrained("outputs/adapter")
tokenizer.save_pretrained("outputs/adapter")
```

#### Axolotl (Config-Driven)

Best for: Multi-GPU, complex training setups, reproducibility across teams.

```bash
pip install axolotl
# Or from source
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl && pip install -e ".[flash-attn]"
```

Example YAML config (`config.yaml`):

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LoraForCausalLM
model_config:
  load_in_4bit: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

datasets:
  - path: ./data/train.jsonl
    type: sharegpt
    field_messages: conversations

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.03
optimizer: adamw_bnb_8bit
bf16: auto

logging_steps: 10
save_steps: 500
output_dir: outputs

flash_attention: true
gradient_checkpointing: true

wandb_project: llm-finetune
```

```bash
axolotl train config.yaml
```

#### LLaMA-Factory (UI + CLI)

Best for: Non-programmers, quick experiments, broad model support.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,bitsandbytes]"
```

```bash
# CLI usage
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset_dir data \
  --dataset my_custom_data \
  --template llama3 \
  --finetuning_type lora \
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_rank 16 \
  --lora_alpha 32 \
  --output_dir outputs \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --bf16 true \
  --logging_steps 10 \
  --save_steps 500 \
  --flash_attn true
```

#### TRL (HuggingFace Native)

Best for: Custom reward functions, novel alignment algorithms, research.

```bash
pip install trl transformers datasets peft accelerate bitsandbytes
```

```python
from trl import SFTTrainer, DPOTrainer, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    optim="adamw_bnb_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=4096,
)
trainer.train()
```

#### TorchTune (PyTorch Native)

Best for: PyTorch-first teams, custom training loops without HF Trainer.

```bash
pip install torchtune
```

```yaml
# sft_lora.yaml
model:
  _component_: torchtune.models.llama3_1.llama3_1_8b
  tokenizer:
    _component_: torchtune.models.llama3.llama3_tokenizer
    path: meta-llama/Llama-3.1-8B-Instruct

checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: meta-llama/Llama-3.1-8B-Instruct
  output_dir: outputs

peft:
  _component_: torchtune.training.peft.LoRA
  lora_attn_modules: ["q_proj", "k_proj", "v_proj", "output_proj"]
  apply_lora_to_mlp: true
  lora_rank: 16
  lora_alpha: 32

dataset:
  _component_: torchtune.datasets.AlpacaDataset

optimizer:
  _component_: torch.optim.AdamW
  lr: 2e-4
  weight_decay: 0.01

lr_scheduler:
  _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
  num_warmup_steps: 100

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  compile: true
```

```bash
tune run lora_finetune_single_device --config sft_lora.yaml
```

#### Raw Transformers (Maximum Control)

Best for: Unusual architectures, debugging, education, full customization.

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    quantization_config=bnb.nn.bitsandbytes_config(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    optim="adamw_bnb_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
model.save_pretrained("outputs/adapter")
```


---

## Model Family Reference

### Llama 2/3/3.1/3.2

```yaml
# config/model-templates.yaml entry
llama3:
  name_or_path: "meta-llama/Meta-Llama-3-8B"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "llama3"
```

**Llama 2 (7B/13B/70B):**
- Chat template: [INST] <</SYS>>{system}<</SYS>>{user} [/INST] {assistant}
- trust_remote_code: false
- Uses GQA (grouped query attention) in 70B
- RoPE scaling not needed for standard context lengths
- 7B: 14GB fp16, ~4GB 4-bit QLoRA

**Llama 3/3.1 (8B/70B/405B):**
- 128K context window (3.1) -- use RoPE scaling for >8K
- tokenizer.chat_template is auto-loaded -- prefer the tokenizer built-in template
- 8B model: 4.5GB in fp16, ~1GB in 4-bit QLoRA

**Llama 3.2 (1B/3B/90B-Vision):**
- Small models (1B/3B) -- ideal for edge deployment
- Vision variant: use LlavaForCausalLM for VLM fine-tuning
- Same chat template as Llama 3.1
- 1B fits easily on T4 even with full fine-tuning

### Qwen 1.5/2/2.5/3

```yaml
qwen2.5:
  name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "qwen"
```

**Qwen 1.5/2 (7B/14B/72B):**
- trust_remote_code: true (for Qwen1.5, some Qwen2 models)
- Qwen2 uses GQA -- 7B has 28 layers, 70B has 80 layers
- Supports very long context (32K-128K)

**Qwen 2.5 (3B/7B/14B/32B/72B):**
- Improved coding and math capabilities
- 32B is a sweet spot for quality vs cost
- 3B model: excellent for edge deployment

**Qwen 3 (0.6B-235B):**
- Latest generation with thinking mode
- Supports hybrid thinking (reason + non-reason)
- Requires Qwen3-specific tokenizer handling
- 0.6B fits on any GPU, 235B needs multi-node

### Mistral/Mixtral

```yaml
mistral:
  name_or_path: "mistralai/Mistral-7B-v0.1"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "mistral"
```

**Mistral 7B (v0.1/v0.2/v0.3):**
- v0.1 uses sliding window attention (4096) but full context for training
- v0.2/v0.3 support 32K context
- Sliding window is not applied during fine-tuning
- 7B: 14GB fp16, ~4GB 4-bit QLoRA

**Mixtral 8x7B / 8x22B:**
- Sparse mixture of experts -- 8 experts, 2 active per token
- 8x7B: 46.7B total params but only ~12.9B active
- Requires trust_remote_code: true
- VRAM: 8x7B needs ~26GB in 4-bit, 8x22B needs ~65GB in 4-bit
- Do NOT use standard LoRA target_modules -- use all MLP experts

### Gemma 1/2/3

```yaml
gemma2:
  name_or_path: "google/gemma-2-9b"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "gemma"
```

**Gemma 1 (2B/7B):**
- 7B: 14GB fp16
- Uses GEGLU activation (gate_proj + up_proj + down_proj)
- Requires accept Google license on HuggingFace

**Gemma 2 (2B/9B/27B):**
- Improved architecture with sliding window + global attention alternating
- 9B: 18GB fp16, ~5GB 4-bit QLoRA
- 9B is a strong mid-range option

**Gemma 3 (1B/4B/12B/27B):**
- Latest generation with multilingual improvements
- 4B: great for edge deployment
- 12B: sweet spot for quality/efficiency
- Supports vision in 12B/27B variants

### Phi 2/3/3.5

```yaml
phi3:
  name_or_path: "microsoft/Phi-3-mini-4k-instruct"
  target_modules: ["qkv_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "phi3"
```

**Phi-2 (2.7B):**
- Very small but surprisingly capable
- Fits on any GPU with full fine-tuning
- 2.7B: 5.4GB fp16

**Phi-3 (3.8B/7B/14B):**
- 3.8B: excellent small model, ~2GB 4-bit
- 14B: competitive with Llama 3 8B
- Note: target_modules uses qkv_proj (fused QKV) instead of separate q/k/v

**Phi-3.5 (3.8B/7B/14B/MoE):**
- MoE variant: 42B total, 6.6B active
- Enhanced reasoning and multilingual

### DeepSeek

```yaml
deepseek-v2:
  name_or_path: "deepseek-ai/DeepSeek-V2-Lite"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "deepseek"
```

- DeepSeek-V2 uses Multi-head Latent Attention (MLA)
- 16B dense model in V2-Lite, 236B MoE in full V2
- trust_remote_code: true required
- For MoE variants, target all expert parameters

### Yi

```yaml
yi:
  name_or_path: "01-ai/Yi-1.5-9B-Chat-16K"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "yi"
```

**Yi-1.5 (6B/9B/34B):**
- 9B: strong multilingual model, 16K context
- trust_remote_code: false
- Uses standard Llama-like architecture
- 9B: 18GB fp16, ~5GB 4-bit QLoRA

### Baichuan

```yaml
baichuan:
  name_or_path: "baichuan-inc/Baichuan2-13B-Chat"
  target_modules: ["W_pack", "gate_proj", "up_proj", "down_proj"]
  chat_template: "baichuan"
```

**Baichuan 2 (7B/13B):**
- trust_remote_code: true (required)
- Uses fused W_pack projection (no separate q/k/v)
- 13B: 26GB fp16, ~7GB 4-bit QLoRA
- Strong Chinese language support

### InternLM

```yaml
internlm:
  name_or_path: "internlm/internlm2_5-7b-chat"
  target_modules: ["qkv_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "internlm"
```

**InternLM 2.5 (7B/20B):**
- 7B: solid general-purpose model
- 20B: competitive with larger models
- trust_remote_code: false
- Uses fused QKV projection
- 7B: 14GB fp16, ~4GB 4-bit QLoRA

### StableLM

```yaml
stablelm:
  name_or_path: "stabilityai/stablelm-3b-4e1t"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  chat_template: "stablelm"
```

**StableLM (3B/7B):**
- 3B: lightweight, good for edge deployment
- 7B: standard mid-range option
- trust_remote_code: false
- 3B: 6GB fp16, ~2GB 4-bit QLoRA
- 7B: 14GB fp16, ~4GB 4-bit QLoRA


---

## Training Method Reference

### SFT (Supervised Fine-Tuning)

The foundation method. Trains the model on (instruction, response) pairs.

**When to use:** Default starting point for all fine-tuning. Domain adaptation, instruction following, style transfer.

**Key hyperparameters:**
- learning_rate: 2e-4 (LoRA) / 2e-5 (full)
- num_epochs: 2-3
- max_seq_length: 4096-8192
- packing: true (faster, but may hurt multi-turn quality)
- warmup_ratio: 0.03
- weight_decay: 0.01

```python
# Unsloth SFT
from unsloth import FastLanguageModel
from trl import SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
    ),
)
trainer.train()
```

### DPO (Direct Preference Optimization)

Alignment method using pairs of preferred/rejected responses. No reward model needed.

**When to use:** After SFT, to align model with human preferences. Requires (chosen, rejected) pairs.

**Key hyperparameters:**
- beta: 0.1 (lower = more divergence from SFT, higher = more conservative)
- learning_rate: 5e-7 to 1e-6 (much lower than SFT)
- num_epochs: 1-3
- max_length: 1024 (prompt + response)
- loss_type: sigmoid (default), hinge, ipo, kto_pair

```python
# TRL DPO
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-7,
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # copy of SFT model
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### KTO (Kahneman-Tversky Optimization)

Alignment from unpaired feedback (thumbs up/down). More practical than DPO.

**When to use:** You have unpaired preferences (not necessarily in pairs). Good when data is noisy.

**Key hyperparameters:**
- beta: 0.1
- learning_rate: 5e-7
- desirable_weight: 1.0 (weight for positive examples)
- undesirable_weight: 1.0 (weight for negative examples)

```python
# TRL KTO
from trl import KTOTrainer, KTOConfig

training_args = KTOConfig(
    output_dir="outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-7,
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

trainer = KTOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,  # with label: True/False
    tokenizer=tokenizer,
)
```

### ORPO (Odds Ratio Preference Optimization)

Single-stage alignment -- combines SFT and preference alignment in one step.

**When to use:** Simpler than DPO (no reference model needed). Good for quick alignment.

**Key hyperparameters:**
- beta: 0.1 (preference strength)
- learning_rate: 1e-5 to 5e-6
- num_epochs: 1-3

```python
# TRL ORPO
from trl import ORPOTrainer, ORPOConfig

training_args = ORPOConfig(
    output_dir="outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    beta=0.1,
)

trainer = ORPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # with chosen/rejected columns
    tokenizer=tokenizer,
)
```

### PPO (Proximal Policy Optimization)

RL alignment with a reward model. Most powerful but most complex.

**When to use:** When you have a trained reward model and want fine-grained control. Research, complex alignment.

**Key hyperparameters:**
- learning_rate: 1e-6
- ppo_epochs: 4
- mini_batch_size: 16
- batch_size: 64
- kl_penalty: "kl" or "abs"
- init_kl_coef: 0.2

```python
# TRL PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
)

config = PPOConfig(
    learning_rate=1e-6,
    ppo_epochs=4,
    mini_batch_size=16,
    batch_size=64,
    kl_penalty="kl",
    init_kl_coef=0.2,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)
```

### GRPO (Group Relative Policy Optimization)

DeepSeek-style alignment. More stable than PPO, no value network needed.

**When to use:** Best alternative to PPO. Simpler, more stable, works well for reasoning tasks.

**Key hyperparameters:**
- learning_rate: 1e-6
- num_generations: 8 (group size)
- max_completion_length: 512
- beta: 0.04 (KL coefficient)
- temperature: 0.7

```python
# TRL GRPO
from trl import GRPOTrainer, GRPOConfig

def reward_func(completions, **kwargs):
    # Custom reward function
    return [1.0 if len(c) > 50 else 0.0 for c in completions]

training_args = GRPOConfig(
    output_dir="outputs",
    per_device_train_batch_size=4,
    num_generations=8,
    max_completion_length=512,
    learning_rate=1e-6,
    beta=0.04,
    temperature=0.7,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_func,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### CPO (Contrastive Preference Optimization)

Single-stage alignment like ORPO but with contrastive loss.

**When to use:** When ORPO gives poor results. Tends to be more stable than ORPO.

**Key hyperparameters:**
- beta: 0.1
- learning_rate: 1e-5
- num_epochs: 1-2

### SimPO (Simple Preference Optimization)

Reference-free DPO variant with length normalization.

**When to use:** When DPO overfits or when you want a simpler setup without reference model.

**Key hyperparameters:**
- beta: 2.0 (higher than DPO)
- gamma: 0.5 (length normalization)
- learning_rate: 5e-7

### Reward Modeling

Train a reward model from preference data for use with PPO.

**When to use:** When you have enough preference data to train a reliable reward signal for PPO.

**Key hyperparameters:**
- learning_rate: 1e-5
- num_epochs: 1-2
- max_length: 1024
- batch_size: 32-64


---

## PEFT Methods Deep Dive

### LoRA (Low-Rank Adaptation)

The default PEFT method. Adds low-rank decomposition matrices to attention layers.

**Parameters:**
- r: Rank (8-64). Higher = more capacity, more VRAM. 16 is the sweet spot.
- alpha: Scaling factor. Usually 2*r. Controls learning rate scaling.
- dropout: 0.05 (default). Can increase to 0.1 for small datasets.
- target_modules: Which layers to adapt. All linear layers is best.

**VRAM impact:** Minimal. LoRA adds ~0.1-0.5GB for a 7B model.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

### QLoRA (Quantized LoRA)

LoRA on a 4-bit quantized base model. Best VRAM efficiency.

**When to use:** When VRAM is limited (Colab/Kaggle T4, consumer GPUs).

**Key settings:**
- 4-bit quantization: NF4 (normalized float 4-bit) is best
- Double quantization: saves ~0.4GB per 7B params
- compute_dtype: bfloat16 or float16

**VRAM savings:** 7B model drops from 14GB (fp16) to ~4GB (QLoRA).

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Decomposes weights into magnitude and direction. Better quality than LoRA.

**When to use:** When LoRA quality is insufficient but full fine-tuning is too expensive.

**Parameters:** Same as LoRA (r, alpha, target_modules).

**VRAM impact:** Slightly more than LoRA (~5-10% overhead).

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_dora=True,  # Enable DoRA
    task_type="CAUSAL_LM",
)
```

### VeRA (Vector-based Random Matrix Adaptation)

Ultra-parameter-efficient. Uses shared random matrices with trainable vectors.

**When to use:** When you need minimal adapter size (mobile, edge deployment).

**Parameters:**
- r: 256 (default, higher than LoRA since it is more efficient)
- lora_alpha: 512
- target_modules: all linear layers

**VRAM impact:** Much less than LoRA (~60% fewer parameters).

### LoHa (Low-Rank Hadamard Product)

Uses Hadamard product of two low-rank matrices. Better expressiveness.

**When to use:** When LoRA underfits. Good for complex tasks.

**Parameters:**
- r: 8-32
- alpha: 2*r
- Similar VRAM to LoRA

### LoKr (Low-Rank Kronecker Product)

Uses Kronecker product of low-rank matrices.

**When to use:** Alternative to LoHa. Slightly different inductive bias.

**Parameters:**
- r: 8-32
- alpha: 2*r

### OFT (Orthogonal Fine-Tuning)

Applies orthogonal transformations. Preserves pretrained knowledge better.

**When to use:** When catastrophic forgetting is a concern. Good for style transfer.

**Parameters:**
- beta: 0.5-1.0 (orthogonality strength)
- target_modules: attention layers
- no_dropout (OFT does not use dropout)

### AdaLoRA (Adaptive LoRA)

Automatically allocates rank to different layers based on importance.

**When to use:** When you do not know which layers to target. Optimizes rank allocation.

**Parameters:**
- target_r: 12 (average rank)
- init_r: 6 (initial rank)
- target_modules: all linear layers

### IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Very lightweight adaptation. Multiplies activations by learned vectors.

**When to use:** Few-shot learning, extreme parameter efficiency.

**Parameters:**
- No rank parameter (it is not low-rank)
- target_modules: ["k_proj", "v_proj", "gate_proj"]
- Very fast training


---

## Platform Setup Guides

### Google Colab (Free T4)

**Session limits:**
- Hard limit: 12 hours
- Idle timeout: ~90 minutes (moves mouse simulation helps)
- GPU: Tesla T4, 15.5GB usable VRAM
- RAM: ~12GB system RAM
- Storage: ~80GB disk (ephemeral)

**Step-by-step setup:**

1. Open Colab, enable GPU: Runtime > Change runtime type > T4 GPU

2. Mount Google Drive for checkpoint persistence:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install dependencies:

```bash
pip install unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

4. Set secrets (API tokens):

```python
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
```

5. Save checkpoints to Drive:

```python
# In TrainingArguments
output_dir="/content/drive/MyDrive/llm-checkpoints"
save_steps=500  # Save frequently - sessions can disconnect
```

6. Auto-reconnect for long training:

```python
# Keep alive - run in a separate cell
import time
while True:
    time.sleep(60)
    # This prevents idle disconnect
```

**Key tips:**
- Save checkpoints every 500 steps minimum
- Use Drive for persistence, not Colab disk
- Download final model before session ends
- Colab Pro (0/mo) gives A100 40GB, longer sessions

### Kaggle (Free T4/P100, 30h/week)

**Session limits:**
- Weekly quota: 30 hours GPU time
- No hard session time limit
- GPU options: T4 (16GB) or P100 (16GB)
- Internet: Enabled by default
- Storage: /kaggle/working (20GB, persists)

**Step-by-step setup:**

1. Create a Kaggle notebook with GPU: Settings > Accelerator > GPU

2. Enable internet: Settings > Internet > On

3. Add a dataset for your training data:
   - Go to Add Data > Your Datasets
   - Upload JSONL/CSV as a dataset
   - Mount in notebook via /kaggle/input/{dataset-name}

4. Set secrets (API tokens):
   - Go to Account > Settings > API > Create New Token
   - In notebook: Add Input > Add Secret > paste token

5. Install dependencies:

```bash
pip install unsloth
```

6. Save outputs to working directory:

```python
# /kaggle/working persists across session restarts
output_dir="/kaggle/working/outputs"
```

7. Push to HuggingFace Hub for persistence:

```python
model.push_to_hub("username/model-name", token=os.environ['HF_TOKEN'])
```

**Key tips:**
- /kaggle/working is the only persistent storage
- /kaggle/input is read-only
- Monitor GPU usage: !nvidia-smi
- P100 is older but sometimes more available than T4
- Sessions can run for hours without idle timeout issues (unlike Colab)

### Local Setup

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA toolkit 11.8+ or 12.x
- Python 3.9+
- 16GB+ RAM recommended

**Step-by-step setup:**

```bash
# Create conda environment
conda create -n llm-train python=3.11
conda activate llm-train

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core packages
pip install transformers datasets peft accelerate trl bitsandbytes
pip install unsloth  # optional but recommended
pip install flash-attn --no-build-isolation  # optional

# Install monitoring
pip install wandb tensorboard

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**CUDA setup notes:**
- PyTorch wheel must match your CUDA version
- Check with: nvidia-smi (look at CUDA Version)
- Flash Attention requires CUDA 11.8+ and specific GPU arch (sm_80+)
- bitsandbytes may need CUDA 11.7+ for some features

### Cloud (RunPod, Lambda, Vast.ai)

**RunPod:**

```bash
# SSH into pod
ssh root@<pod-ip> -p <port>

# Setup
pip install unsloth transformers datasets peft accelerate

# Quick test
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

- GPU options: A10 24GB, A100 40/80GB, H100 80GB
- Pricing: $0.20-$3.50/hr depending on GPU
- Community cloud is cheapest
- Persistent storage via network volumes

**Lambda Labs:**

```bash
# SSH into instance
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>

# Setup
sudo apt update && sudo apt install -y python3-pip
pip3 install unsloth transformers datasets peft accelerate
```

- GPU options: A10 24GB ($0.60/hr), A100 40GB ($1.10/hr), A100 80GB ($2.64/hr)
- 1-click templates available
- Good for longer training runs

**Vast.ai:**

```bash
# Search for cheap GPUs
vast search offers 'gpu_name=RTX_3090 num_gpus=1 dph_total<0.30'

# SSH into instance
ssh root@<instance-ip> -p <port>
```

- Cheapest option for consumer GPUs
- RTX 3090: $0.10-$0.30/hr
- Inconsistent availability, quality varies
- Good for non-critical training, experimentation

---

## Dataset Preparation

### Supported Formats

**ShareGPT (Preferred):**

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is machine learning?"},
      {"from": "gpt", "value": "Machine learning is a subset of AI..."},
      {"from": "human", "value": "Can you give an example?"},
      {"from": "gpt", "value": "Sure, here is an example..."}
    ]
  }
]
```

**Alpaca:**

```json
[
  {
    "instruction": "Explain quantum computing",
    "input": "in simple terms",
    "output": "Quantum computing uses qubits that can be 0 and 1 simultaneously..."
  }
]
```

**OpenAI:**

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hello! How can I help?"}
    ]
  }
]
```

### Format Conversion

Use `scripts/prepare_dataset.py` to convert between formats:

```bash
# Convert Alpaca to ShareGPT
python scripts/prepare_dataset.py --input data/alpaca.json --output data/sharegpt.json --format alpaca

# ShareGPT passthrough
python scripts/prepare_dataset.py --input data/sharegpt.json --output data/train.jsonl --format sharegpt
```

### Multi-turn Chat Datasets

For multi-turn conversations, ensure the dataset has alternating human/gpt turns:

```json
{
  "conversations": [
    {"from": "human", "value": "What is Python?"},
    {"from": "gpt", "value": "Python is a programming language..."},
    {"from": "human", "value": "Is it good for AI?"},
    {"from": "gpt", "value": "Yes, Python is the most popular language for AI..."}
  ]
}
```

### Tool Calling Datasets

For tool/function calling fine-tuning, include tool definitions and structured outputs:

```json
{
  "conversations": [
    {"from": "human", "value": "What is the weather in NYC?"},
    {"from": "gpt", "value": "Let me check the weather for you.", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}]},
    {"from": "human", "value": "The weather is 72F and sunny."},
    {"from": "gpt", "value": "The weather in NYC is currently 72F and sunny."}
  ]
}
```

### Dataset Validation

Always validate your dataset before training:

```python
from datasets import load_dataset
import json

# Load and inspect
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
print(f"Total samples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"Sample: {dataset[0]}")

# Check for empty conversations
empty = sum(1 for d in dataset if len(str(d)) < 10)
print(f"Empty samples: {empty}")
```

### Train/Val Split

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
split = dataset.train_test_split(test_size=0.05, seed=42)
split["train"].to_json("data/train_split.jsonl")
split["test"].to_json("data/val_split.jsonl")
```

---

## VRAM Estimation

### Using estimate_vram.py

Use `scripts/estimate_vram.py` to check if your training configuration fits in VRAM:

```bash
# Basic estimation
python scripts/estimate_vram.py --params-b 8 --seq-len 4096 --batch-size 2 --quantize 4bit

# Example output:
# Estimated VRAM: 7.83 GB
# Feasible on T4: True
# Feasible on RTX 3090: True
# Feasible on A100: True
```

### Parameters

- `--params-b`: Model size in billions (e.g., 7, 8, 13, 70)
- `--seq-len`: Maximum sequence length (2048, 4096, 8192)
- `--batch-size`: Per-device batch size
- `--quantize`: Quantization level (none, 4bit, 8bit)

### How Estimation Works

The estimator accounts for:
1. **Model weights**: Parameters * bytes_per_param (fp16=2, 8bit=1, 4bit=0.5)
2. **Activations**: Proportional to model size * seq_len * batch_size
3. **Optimizer states**: LoRA adds ~1.5GB overhead; full fine-tuning adds 4x model params
4. **CUDA context**: ~1.5GB fixed overhead

### Optimization Strategies When VRAM Is Tight

1. **Reduce batch size, increase gradient accumulation**: Same effective batch, less VRAM
2. **Enable QLoRA (4-bit)**: Drops model weight memory by 4x
3. **Reduce max_seq_length**: Halving seq_len roughly halves activation memory
4. **Enable gradient checkpointing**: Trades compute for ~30% less activation memory
5. **Use packing**: Packs multiple short sequences, reduces padding waste
6. **Enable Flash Attention 2**: Reduces attention memory from O(n^2) to O(n)
7. **CPU offloading**: Moves optimizer states to CPU (slow but saves VRAM)

### VRAM Reference Table

| Config | Model | Quant | Seq | Batch | Est. VRAM | Fits |
|--------|-------|-------|-----|-------|-----------|------|
| QLoRA 8B | 8B | 4bit | 4096 | 2 | ~7.8 GB | T4, RTX 3090, A100 |
| QLoRA 8B | 8B | 4bit | 8192 | 4 | ~12.5 GB | T4 (tight), RTX 3090, A100 |
| Full 8B | 8B | fp16 | 4096 | 2 | ~22 GB | RTX 3090, A100 |
| QLoRA 13B | 13B | 4bit | 4096 | 2 | ~10.5 GB | RTX 3090, A100 |
| QLoRA 70B | 70B | 4bit | 4096 | 1 | ~38 GB | A100-80, H100 |

---

## Optimization Techniques

### Flash Attention 2

Reduces attention memory from O(n^2) to O(n) and speeds up computation.

```bash
# Install
pip install flash-attn --no-build-isolation

# Requires: CUDA 11.8+, GPU arch sm_80+ (A100, RTX 3090/4090, H100)
# T4 does NOT support Flash Attention 2
```

```python
# In TrainingArguments
args = TrainingArguments(
    attn_implementation="flash_attention_2",  # or "sdpa"
    ...
)
```

**When to use:** Always enable if your GPU supports it (A100, RTX 3090/4090, H100). Not available on T4.

### SDPA (Scaled Dot Product Attention)

PyTorch native attention optimization. Works on more GPUs than Flash Attention 2.

```python
args = TrainingArguments(
    attn_implementation="sdpa",
    ...
)
```

**When to use:** Fallback when Flash Attention 2 is not available (T4, P100, V100).

### Gradient Checkpointing

Trades compute for memory by recomputing activations during backward pass.

```python
args = TrainingArguments(
    gradient_checkpointing=True,
    ...
)
```

**Memory savings:** ~30-40% less activation memory. ~20% slower training.

### Activation and CPU Offloading

Moves optimizer states or activations to CPU to save VRAM.

```python
# CPU Offloading via DeepSpeed
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    }
}
```

**When to use:** When VRAM is extremely tight and you need to fit a larger model. Significant speed penalty (2-3x slower).

### 8-bit and 4-bit Adam Optimizer

Reduces optimizer state memory by 50-75%.

```python
# 8-bit Adam
args = TrainingArguments(
    optim="adamw_bnb_8bit",
    ...
)

# Or using adafactor (no optimizer states)
args = TrainingArguments(
    optim="adafactor",
    ...
)
```

**When to use:** Always recommended for QLoRA training. adafactor is a good alternative for full fine-tuning.

### GaLore (Gradient Low-Rank Projection)

Reduces optimizer memory by projecting gradients to low-rank space.

```python
from galore import GaLoreAdamW

optimizer = GaLoreAdamW(model.parameters(), lr=2e-4, rank=128)
```

**Memory savings:** 60-80% less optimizer memory. Requires custom optimizer setup.

### Packing

Combines multiple short sequences into one batch to reduce padding waste.

```python
# Unsloth
trainer = SFTTrainer(
    ...
    packing=True,  # Enable packing
)

# Axolotl
sample_packing: true
```

**When to use:** When your dataset has many short sequences. Can hurt multi-turn conversation quality since conversations may be mixed.

### LoRA+

Uses different learning rates for LoRA matrices A and B.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    loq_lr=1e-5,  # Learning rate for matrix A
    ...
)
```

**When to use:** Can improve LoRA convergence by 1-3% on some benchmarks.

### LoRA-GA (LoRA Gradient Approximation)

Better initialization for LoRA using gradient information.

```python
from peft import LoraConfig, LoraModel

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    init_lora_weights="loftq",  # Uses LoRA-GA initialization
    ...
)
```

**When to use:** When standard LoRA convergence is slow. Better initialization = faster convergence.

---

## Multi-GPU Training

### DDP (Distributed Data Parallel)

Simplest multi-GPU strategy. Each GPU has a full model copy.

```bash
# Launch with accelerate
accelerate launch --num_processes 2 train.py

# Or with torchrun
torchrun --nproc_per_node 2 train.py
```

**When to use:** Models that fit in single GPU VRAM. Simple, fast, good scaling.

### FSDP (Fully Sharded Data Parallel)

Shards model parameters, gradients, and optimizer states across GPUs.

```bash
# Launch with accelerate
accelerate launch --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer train.py

# Or with torchrun + DeepSpeed
torchrun --nproc_per_node 2 train.py --deepspeed ds_config_zero3.json
```

**When to use:** Models that do not fit on a single GPU. Good for 70B+ models.

### DeepSpeed ZeRO

Three stages of memory optimization:

**ZeRO Stage 1:** Shards optimizer states only
```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```

**ZeRO Stage 2:** Shards optimizer states + gradients
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"}
    }
}
```

**ZeRO Stage 3:** Shards everything (parameters, gradients, optimizer states)
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"}
    }
}
```

**When to use which:**
- Stage 1: 2-4 GPUs, model fits mostly in single GPU
- Stage 2: 2-8 GPUs, model is slightly too large for single GPU
- Stage 3: 4-8+ GPUs, model is much larger than single GPU
- With CPU offloading: Maximum memory savings, significant speed penalty

### Multi-GPU Config Examples

**Axolotl FSDP:**

```yaml
# In Axolotl config
micro_batch_size: 2
gradient_accumulation_steps: 4
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

**DeepSpeed ZeRO-3 for 70B:**

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": false
}
```

---

## Export Pipeline

### Step 1: Merge LoRA Adapters

After training, merge the LoRA adapter back into the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Load and merge adapter
model = PeftModel.from_pretrained(base_model, "outputs/adapter")
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("outputs/merged")
tokenizer = AutoTokenizer.from_pretrained("outputs/adapter")
tokenizer.save_pretrained("outputs/merged")
```

Or use the merge script:

```bash
python scripts/merge_lora.py --base meta-llama/Llama-3.1-8B-Instruct --adapter outputs/adapter --output outputs/merged
```

### Step 2: Convert to GGUF

For Ollama, LM Studio, llama.cpp deployment:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert to GGUF
python convert_hf_to_gguf.py ../outputs/merged --outfile ../outputs/model-f16.gguf --outtype f16

# Quantize
./llama-quantize ../outputs/model-f16.gguf ../outputs/model-Q4_K_M.gguf Q4_K_M
./llama-quantize ../outputs/model-f16.gguf ../outputs/model-Q5_K_M.gguf Q5_K_M
./llama-quantize ../outputs/model-f16.gguf ../outputs/model-Q8_0.gguf Q8_0
```

**GGUF quantization levels:**
- Q4_K_M: 4-bit, good balance of size and quality (recommended for most use)
- Q5_K_M: 5-bit, slightly better quality, slightly larger
- Q8_0: 8-bit, near-original quality, larger file
- F16: Full 16-bit, no quality loss, same size as merged model

Or use the script:

```bash
python scripts/convert_to_gguf.py --model outputs/merged --output outputs/model --quants Q4_K_M Q5_K_M Q8_0
```

### Step 3: Convert to AWQ

For vLLM, TGI deployment with quantized models:

```bash
pip install autoawq

python -m awq.entry --model_path outputs/merged     --w_bit 4     --q_group_size 128     --calib_data "your-dataset"     --output_path outputs/model-awq
```

Or use the script:

```bash
python scripts/convert_to_awq.py --model outputs/merged --output outputs/model-awq --bits 4
```

### Step 4: Convert to GPTQ

For AutoGPTQ, TGI deployment:

```bash
pip install auto-gptq

python -m auto_gptq.modeling.eval --model outputs/merged     --quantize     --bits 4     --group-size 128     --output outputs/model-gptq
```

Or use the script:

```bash
python scripts/convert_to_gptq.py --model outputs/merged --output outputs/model-gptq --bits 4
```

### Deployment Compatibility

| Format | Ollama | LM Studio | vLLM | TGI | llama.cpp |
|--------|--------|-----------|------|-----|-----------|
| GGUF | Yes | Yes | No | No | Yes |
| AWQ | No | No | Yes | Yes | No |
| GPTQ | No | No | Yes | Yes | No |
| HF safetensors | Yes | Yes | Yes | Yes | Yes |

### Push to HuggingFace Hub

```bash
python scripts/push_to_hub.py --model outputs/merged --repo username/model-name --private
```

Or manually:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="outputs/merged", repo_id="username/model-name", repo_type="model")
```

---

## Monitoring and Logging

### Weights and Biases (W&B)

```bash
pip install wandb
wandb login  # Enter your API key
```

```python
# In TrainingArguments
args = TrainingArguments(
    report_to="wandb",
    project_name="llm-finetune",
    run_name="llama31-8b-qlora-sft",
    logging_steps=10,
    ...
)
```

```python
# Or set via environment variable
import os
os.environ["WANDB_PROJECT"] = "llm-finetune"
os.environ["WANDB_NAME"] = "llama31-8b-qlora-sft"
```

### TensorBoard

```bash
pip install tensorboard
```

```python
args = TrainingArguments(
    report_to="tensorboard",
    logging_dir="./logs",
    logging_steps=10,
    ...
)
```

```bash
# Launch TensorBoard
tensorboard --logdir ./logs
# Opens at http://localhost:6006
```

### MLflow

```bash
pip install mlflow
```

```python
args = TrainingArguments(
    report_to="mlflow",
    ...
)

# Log manually
import mlflow
mlflow.log_metric("eval_loss", 0.15)
mlflow.log_param("learning_rate", 2e-4)
```

### Key Metrics to Watch

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| Training loss | Decreasing steadily | Plateaus early, NaN, spikes |
| Learning rate | Follows schedule | Stuck at one value |
| Gradient norm | < 10 | > 100 = explosion |
| GPU memory | < 90% utilization | > 95% = OOM risk |
| Epoch time | Consistent | Increasing = memory leak |
| Eval loss | Decreasing | Increasing = overfitting |

---

## Diagnostic Playbook

### OOM (Out of Memory) Errors

**Symptoms:** `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solutions by GPU tier:**

**T4 (16GB):**
1. Switch to QLoRA (4-bit): `load_in_4bit=True`
2. Reduce batch size to 1, increase gradient accumulation
3. Reduce max_seq_length to 2048
4. Enable gradient checkpointing
5. Use only q_proj, v_proj as target_modules

**RTX 3090/4090 (24GB):**
1. Try QLoRA if full fine-tuning OOMs
2. Reduce batch size from 4 to 2
3. Enable gradient checkpointing
4. Reduce max_seq_length from 8192 to 4096

**A100 (40/80GB):**
1. Reduce batch size
2. Enable DeepSpeed ZeRO Stage 2/3 for 70B+ models
3. Use CPU offloading for very large models

**General OOM fixes:**
```python
# Reduce memory usage
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    max_seq_length=2048,
    optim="adamw_bnb_8bit",
    ...
)
```

### NaN Losses

**Causes and fixes:**

1. **Learning rate too high:** Reduce by 10x (2e-4 -> 2e-5)
2. **Mixed precision issues:** Try fp16 instead of bf16, or vice versa
3. **Exploding gradients:** Add `max_grad_norm=1.0`
4. **Corrupted data:** Remove samples with empty/very long responses
5. **Tokenizer issues:** Ensure pad_token is set: `tokenizer.pad_token = tokenizer.eos_token`

```python
# Fix NaN losses
args = TrainingArguments(
    learning_rate=2e-5,  # Lower LR
    max_grad_norm=1.0,   # Gradient clipping
    fp16=True,           # Or bf16=True
    ...
)
```

### Convergence Issues

**Symptoms:** Loss not decreasing, or decreasing very slowly.

**Checklist:**
1. **Learning rate:** Try 2e-4 for LoRA, 2e-5 for full fine-tuning
2. **Data quality:** Inspect samples manually, remove noise
3. **Data quantity:** Need at least 1000 samples for meaningful fine-tuning
4. **Epochs:** Try 3 epochs (too few = underfit, too many = overfit)
5. **Batch size:** Effective batch size of 16-32 is typical sweet spot
6. **Chat template:** Ensure correct template for model family
7. **Sequence length:** Too short truncates responses, too long wastes compute
8. **PEFT rank:** Try r=32 if r=16 underfits

### Chat Template Mismatch

**Symptoms:** Model generates garbled output, ignores system prompt, poor quality.

**Fix:** Always use the model's native chat template:

```python
# Best approach - use tokenizer's built-in template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# tokenizer.chat_template is auto-loaded

# Manual template for Llama 3.1
template = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)
```

### Tokenizer Issues

**Common problems and fixes:**

1. **Missing pad_token:** `tokenizer.pad_token = tokenizer.eos_token`
2. **Wrong vocab size:** Check `len(tokenizer)` matches model config
3. **Special tokens:** Ensure special tokens are added correctly

```python
# Verify tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(f"Vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"Chat template exists: {tokenizer.chat_template is not None}")
```

### Gradient Explosion

**Symptoms:** Loss spikes to very large values, NaN after spike.

**Fixes:**
```python
args = TrainingArguments(
    max_grad_norm=1.0,  # Gradient clipping
    learning_rate=2e-5,  # Lower LR
    warmup_ratio=0.03,   # Warmup helps stability
    ...
)
```

### Training Hangs

**Causes:**
1. **Data loading deadlock:** Set `dataloader_num_workers=0` or `dataloader_num_workers=2`
2. **NCCL timeout:** Set `NCCL_BLOCKING_WAIT=1`
3. **Dead GPU:** Check `nvidia-smi` for GPU status

**Fix:**
```python
args = TrainingArguments(
    dataloader_num_workers=2,
    ddp_timeout=1800,  # 30 min timeout
    ...
)
```

---

## Hardware Reference

### VRAM by GPU

| GPU | VRAM | FP16 | QLoRA 4-bit | Full FT 8B | QLoRA 70B |
|-----|------|------|-------------|------------|-----------|
| T4 | 16GB | 8B | 8B | 3B | 7B (tight) |
| P100 | 16GB | 8B | 8B | 3B | 7B (tight) |
| RTX 3090 | 24GB | 13B | 13B | 8B | 13B |
| RTX 4090 | 24GB | 13B | 13B | 8B | 13B |
| A10G | 24GB | 13B | 13B | 8B | 13B |
| A100-40 | 40GB | 34B | 34B | 13B | 34B |
| A100-80 | 80GB | 70B | 70B | 34B | 70B |
| H100 | 80GB | 70B | 70B | 34B | 70B |

### Max Model Size by GPU

| GPU | QLoRA | Full Fine-tune |
|-----|-------|----------------|
| T4 (16GB) | 7B-8B | 1B-3B |
| RTX 3090 (24GB) | 13B | 7B-8B |
| A100-40 (40GB) | 34B | 13B |
| A100-80 (80GB) | 70B | 34B |
| H100 (80GB) | 70B | 34B |

### Batch Size Recommendations

| GPU | 8B QLoRA | 13B QLoRA | 70B QLoRA |
|-----|----------|-----------|-----------|
| T4 | 2 | 1 | N/A |
| RTX 3090 | 4 | 2 | 1 |
| A100-40 | 8 | 4 | 2 |
| A100-80 | 16 | 8 | 4 |

---

## Cost Optimization

### Free Tier Strategies

**Google Colab:**
- Session limit: 12 hours, idle timeout ~90 min
- Save checkpoints every 500 steps to Drive
- Use auto-reconnect cell
- Colab Pro: $10/mo for A100 40GB

**Kaggle:**
- 30 hours/week GPU quota
- T4 or P100 available
- No idle timeout (better than Colab)
- Save to /kaggle/working (persistent)

**Combined strategy:**
- Start with Kaggle (more GPU hours)
- Use Colab for quick experiments
- Switch to cloud for large models

### Checkpoint Frequency

```python
# For 12-hour Colab sessions
save_steps = 500  # Save every 500 steps
# For 30-hour Kaggle sessions
save_steps = 1000  # Can save less frequently

# Always save to persistent storage
output_dir = "/content/drive/MyDrive/llm-checkpoints"  # Colab
output_dir = "/kaggle/working/outputs"  # Kaggle
```

### Mixed Precision Benefits

| Precision | VRAM | Speed | Quality |
|-----------|------|-------|---------|
| FP32 | Baseline | Baseline | Baseline |
| FP16 | 50% less | 2x faster | Same |
| BF16 | 50% less | 2x faster | Same (better range) |

**Always use bf16 if your GPU supports it** (A100, H100, RTX 3090/4090). Use fp16 for T4/P100.

### Gradient Accumulation

Effective batch size = per_device_batch_size * gradient_accumulation_steps * num_gpus

```python
# Instead of batch_size=32 (needs lots of VRAM):
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # Effective batch size = 2 * 16 = 32
    ...
)
```

### Training Time Estimates

| Config | Model | GPU | Dataset | Time |
|--------|-------|-----|---------|------|
| QLoRA SFT | 8B | T4 | 10K samples | ~2 hours |
| QLoRA SFT | 8B | RTX 3090 | 10K samples | ~45 min |
| QLoRA DPO | 8B | T4 | 5K pairs | ~1.5 hours |
| Full SFT | 8B | A100-80 | 10K samples | ~4 hours |
| QLoRA SFT | 70B | A100-80 | 10K samples | ~6 hours |

---

## Scripts Reference

### prepare_dataset.py

Converts and validates datasets for training.

```bash
# Convert Alpaca to ShareGPT
python scripts/prepare_dataset.py --input data/alpaca.json --output data/sharegpt.json --format alpaca

# ShareGPT passthrough
python scripts/prepare_dataset.py --input data/raw.json --output data/train.jsonl --format sharegpt
```

### estimate_vram.py

Estimates VRAM requirements for a given configuration.

```bash
python scripts/estimate_vram.py --params-b 8 --seq-len 4096 --batch-size 2 --quantize 4bit
```

### launch_training.py

Unified launcher with platform detection and crash recovery.

```bash
# Local
python scripts/launch_training.py --config config.yaml

# Colab/Kaggle (auto-detected)
python scripts/launch_training.py --config config.yaml --platform colab
```

### merge_lora.py

Merges LoRA adapters into base model.

```bash
python scripts/merge_lora.py --base meta-llama/Llama-3.1-8B-Instruct --adapter outputs/adapter --output outputs/merged
```

### convert_to_gguf.py

Converts HF model to GGUF with quantization.

```bash
python scripts/convert_to_gguf.py --model outputs/merged --output outputs/model --quants Q4_K_M Q5_K_M Q8_0
```

### convert_to_awq.py

Quantizes model to AWQ format.

```bash
python scripts/convert_to_awq.py --model outputs/merged --output outputs/model-awq --bits 4
```

### convert_to_gptq.py

Quantizes model to GPTQ format.

```bash
python scripts/convert_to_gptq.py --model outputs/merged --output outputs/model-gptq --bits 4
```

### eval_model.py

Evaluates model on benchmarks and custom datasets.

```bash
python scripts/eval_model.py --model outputs/merged --benchmark perplexity --dataset wikitext-2
```

### push_to_hub.py

Uploads model to HuggingFace Hub.

```bash
python scripts/push_to_hub.py --model outputs/merged --repo username/model-name --private
```

### colab_setup.py

Colab-specific environment setup.

```python
# Run at start of Colab notebook
exec(open('scripts/colab_setup.py').read())
```

### kaggle_setup.py

Kaggle-specific environment setup.

```python
# Run at start of Kaggle notebook
exec(open('scripts/kaggle_setup.py').read())
```

### diagnose_crash.py

Analyzes training logs for OOM, NaN, and hang issues.

```bash
python scripts/diagnose_crash.py --log outputs/trainer_state.json
```

### monitor_training.py

Parses training logs, detects anomalies, supports early stopping.

```bash
python scripts/monitor_training.py --log-dir outputs --early-stop-patience 5
```
