from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
import torch

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "data/train.jsonl"
OUTPUT_DIR = "outputs/trl-sft-lora"
MAX_SEQ_LENGTH = 4096
PACKING = True

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
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
RUN_NAME = "trl-llama31-8b-sft-lora"

# ============================================
# MAIN
# ============================================

def formatting_func(example):
    conversations = example["conversations"]
    text = ""
    for conv in conversations:
        if conv["from"] == "human":
            text += f"<|user|>\n{conv['value']}<|end|>\n"
        elif conv["from"] == "gpt":
            text += f"<|assistant|>\n{conv['value']}<|end|>\n"
    return text

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# LoRA
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM",
)

# Dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(lambda x: {"text": formatting_func(x)})

# Trainer
training_args = TrainingArguments(
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
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=PACKING,
)

trainer.train()

# Save
model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")