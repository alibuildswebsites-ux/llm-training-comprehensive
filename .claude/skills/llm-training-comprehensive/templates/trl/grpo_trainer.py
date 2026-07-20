from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SFT_MODEL_PATH = "outputs/sft-model"  # Path to SFT model
DATASET_PATH = "data/grpo_train.jsonl"  # GRPO format: prompt, responses
OUTPUT_DIR = "outputs/trl-grpo-lora"
MAX_COMPLETION_LENGTH = 512

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
PER_DEVICE_BATCH_SIZE = 4
NUM_GENERATIONS = 8  # Group size
NUM_EPOCHS = 1
LEARNING_RATE = 1e-6
BETA = 0.04
TEMPERATURE = 0.7
BF16 = True
LOGGING_STEPS = 10
OPTIM = "adamw_torch_fused"
SEED = 42
SAVE_STEPS = 500
REPORT_TO = "wandb"
RUN_NAME = "trl-llama31-8b-grpo-lora"

# ============================================
# REWARD FUNCTIONS
# ============================================

def reward_correctness(completions, **kwargs):
    """Verify answer correctness"""
    rewards = []
    for completion in completions:
        # Extract answer from completion
        # Custom logic based on your task
        # Example: check if answer matches expected
        rewards.append(1.0 if is_correct(completion) else 0.0)
    return rewards

def reward_format(completions, **kwargs):
    """Reward proper formatting"""
    rewards = []
    for completion in completions:
        # Check for required format (e.g., <answer>...</answer>)
        rewards.append(1.0 if has_correct_format(completion) else 0.0)
    return rewards

def is_correct(completion):
    # Implement your correctness check
    return True  # Placeholder

def has_correct_format(completion):
    # Implement your format check
    return True  # Placeholder

# ============================================
# MAIN
# ============================================

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    beta=BETA,
    temperature=TEMPERATURE,
    bf16=BF16,
    logging_steps=LOGGING_STEPS,
    optim=OPTIM,
    seed=SEED,
    save_steps=SAVE_STEPS,
    report_to=REPORT_TO,
    run_name=RUN_NAME,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=[reward_correctness, reward_format],
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")