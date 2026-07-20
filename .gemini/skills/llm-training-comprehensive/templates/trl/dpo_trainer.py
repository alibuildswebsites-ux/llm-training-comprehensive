from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SFT_MODEL_PATH = "outputs/sft-model"  # Path to SFT model
DATASET_PATH = "data/dpo_train.jsonl"  # DPO format: prompt, chosen, rejected
OUTPUT_DIR = "outputs/trl-dpo-lora"
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-7
BETA = 0.1
LOSS_TYPE = "sigmoid"
FP16 = not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_bf16_supported()
LOGGING_STEPS = 10
OPTIM = "adamw_bnb_8bit"
SEED = 42
SAVE_STEPS = 500
REPORT_TO = "wandb"
RUN_NAME = "trl-llama31-8b-dpo-lora"

# ============================================
# MAIN
# ============================================

# Policy model (from SFT)
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
)

# Reference model (frozen copy of SFT)
ref_model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
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

model = get_peft_model(model, peft_config)

# Dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Training args
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    beta=BETA,
    loss_type=LOSS_TYPE,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fp16=FP16,
    bf16=BF16,
    logging_steps=LOGGING_STEPS,
    optim=OPTIM,
    seed=SEED,
    save_steps=SAVE_STEPS,
    report_to=REPORT_TO,
    run_name=RUN_NAME,
    gradient_checkpointing=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")