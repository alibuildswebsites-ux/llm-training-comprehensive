from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# DPO specific
BETA = 0.1
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# Data
DATASET_PATH = "data/dpo_train.jsonl"  # DPO format: prompt, chosen, rejected
DATASET_TEXT_FIELD = "text"

# Training
OUTPUT_DIR = "outputs/unsloth-dpo-lora"
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LOGGING_STEPS = 10
OPTIM = "adamw_bnb_8bit"
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
SEED = 42
SAVE_STEPS = 500
REPORT_TO = "wandb"
RUN_NAME = "llama31-8b-dpo-lora"

# ============================================
# MAIN
# ============================================

def formatting_func(example):
    """Format DPO data for Unsloth"""
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Load model (SFT checkpoint or base)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_func)

# DPO Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Unsloth handles this
    args=DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        loss_type="sigmoid",
        logging_steps=LOGGING_STEPS,
        optim=OPTIM,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        save_steps=SAVE_STEPS,
        report_to=REPORT_TO,
        run_name=RUN_NAME,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")

print("DPO training complete!")