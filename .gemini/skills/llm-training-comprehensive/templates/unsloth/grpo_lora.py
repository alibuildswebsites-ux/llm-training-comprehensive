from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
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

# GRPO specific
LEARNING_RATE = 1e-6
NUM_GENERATIONS = 8
MAX_COMPLETION_LENGTH = 512
BETA = 0.04
TEMPERATURE = 0.7

# Data
DATASET_PATH = "data/grpo_train.jsonl"  # Prompt + reward function
OUTPUT_DIR = "outputs/unsloth-grpo-lora"

# Training
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 1
LOGGING_STEPS = 10
OPTIM = "adamw_torch_fused"
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
SEED = 42
SAVE_STEPS = 500
REPORT_TO = "wandb"
RUN_NAME = "llama31-8b-grpo-lora"

# ============================================
# REWARD FUNCTION
# ============================================

def reward_func(completions, **kwargs):
    """
    Custom reward function for GRPO.
    Returns list of rewards (floats) for each completion.
    
    Examples:
    - Math: correctness of answer
    - Code: passes tests
    - Format: follows required structure
    - Length: penalize too short/long
    """
    rewards = []
    for completion in completions:
        # Example: extract answer and check
        text = completion[0]["content"] if isinstance(completion, list) else completion
        
        # Your reward logic here
        reward = 1.0 if len(text) > 50 else 0.0  # Placeholder
        rewards.append(reward)
    
    return rewards

# ============================================
# MAIN
# ============================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

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

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    beta=BETA,
    temperature=TEMPERATURE,
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
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_func,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")

print("GRPO training complete!")