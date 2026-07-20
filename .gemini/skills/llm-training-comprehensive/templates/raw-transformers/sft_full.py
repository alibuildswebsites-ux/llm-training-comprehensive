import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ============================================
# CONFIGURATION - Full Fine-Tuning
# ============================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096

# Training (full FT needs more memory, lower LR)
OUTPUT_DIR = "outputs/raw-sft-full"
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
WARMUP_RATIO = 0.03
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5  # Lower for full FT
BF16 = True
FP16 = False
LOGGING_STEPS = 10
OPTIM = "adamw_torch_fused"
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
REPORT_TO = "wandb"
RUN_NAME = "llama31-8b-raw-sft-full"

# Data
DATASET_PATH = "data/train.jsonl"
TEXT_FIELD = "text"

# ============================================
# MAIN
# ============================================

def formatting_func(example):
    messages = example["conversations"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {TEXT_FIELD: text}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model (FULL PRECISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    device_map="auto",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_func, batched=False)

def tokenize(examples):
    return tokenizer(
        examples[TEXT_FIELD],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    bf16=BF16,
    fp16=FP16,
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
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/model")

print("Full fine-tuning complete!")