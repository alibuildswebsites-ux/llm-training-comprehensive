from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# Full fine-tuning (no LoRA)
# Note: Requires significantly more VRAM
# For 8B full FT: ~22GB (fp16), ~12GB (bf16 with gradient checkpointing)

# Data
DATASET_PATH = "data/sft_train.jsonl"
DATASET_TEXT_FIELD = "text"
PACKING = True

# Training
OUTPUT_DIR = "outputs/unsloth-sft-full"
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
WARMUP_RATIO = 0.03
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5  # Lower for full FT
FP16 = not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_bf16_supported()
LOGGING_STEPS = 10
OPTIM = "adamw_torch_fused"
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
REPORT_TO = "wandb"
RUN_NAME = "llama31-8b-sft-full"

# ============================================
# MAIN
# ============================================

def formatting_func(example):
    """Format conversations to text"""
    messages = example["conversations"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# Load model (NO LoRA - full fine-tuning)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,  # Full precision for full FT
)

# Prepare for k-bit training (if using quantization)
# model = FastLanguageModel.prepare_model_for_kbit_training(model)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_func, batched=False)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field=DATASET_TEXT_FIELD,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=PACKING,
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
        gradient_checkpointing=True,
    ),
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/model")

print("Full fine-tuning complete!")