from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================
# CONFIGURATION - Modify these for your run
# ============================================

# Model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
DTYPE = None  # Auto-detect

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
USE_GRADIENT_CHECKPOINTING = "unsloth"

# Data
DATASET_PATH = "data/train.jsonl"  # ShareGPT format
DATASET_TEXT_FIELD = "text"
PACKING = True

# Training
OUTPUT_DIR = "outputs/unsloth-sft-lora"
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.03
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
FP16 = not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_bf16_supported()
LOGGING_STEPS = 10
OPTIM = "adamw_bnb_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
REPORT_TO = "wandb"
RUN_NAME = "llama31-8b-sft-lora"

# ============================================
# MAIN TRAINING CODE
# ============================================

def formatting_func(example):
    """Format ShareGPT conversations to text"""
    conversations = example["conversations"]
    text = ""
    for conv in conversations:
        if conv["from"] == "human":
            text += f"<|user|>\n{conv['value']}<|end|>\n"
        elif conv["from"] == "gpt":
            text += f"<|assistant|>\n{conv['value']}<|end|>\n"
    return text

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
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
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(lambda x: {"text": formatting_func(x)})

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field=DATASET_TEXT_FIELD,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=PACKING,
    formatting_func=formatting_func,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=REPORT_TO,
        run_name=RUN_NAME,
    ),
)

# Train
trainer.train()

# Save adapter
model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")

print("Training complete! Adapter saved to:", f"{OUTPUT_DIR}/adapter")