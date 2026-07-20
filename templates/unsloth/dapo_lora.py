from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.9,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load dataset
dataset = load_dataset("trl-lib/tldr", split="train[:2000]")

# DAPO-specific reward functions
def reward_format(completions, **kwargs):
    rewards = []
    for c in completions:
        if "```" in c:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_length(completions, **kwargs):
    """DAPO uses length-normalized rewards"""
    rewards = []
    for c in completions:
        length = len(c.split())
        # Normalize by length (DAPO key innovation)
        if 100 <= length <= 800:
            rewards.append(1.0)
        elif 50 <= length < 100 or 800 < length <= 1200:
            rewards.append(0.5)
        else:
            rewards.append(0.1)
    return rewards

def reward_correctness(completions, prompts, **kwargs):
    """Verify correctness for math/code tasks"""
    rewards = []
    for c, p in zip(completions, prompts):
        # Extract answer from completion
        # This is task-specific; implement your verifier
        rewards.append(1.0)  # Placeholder
    return rewards

# Train DAPO (uses GRPOTrainer with DAPO config)
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[reward_format, reward_length, reward_correctness],
    args=GRPOConfig(
        output_dir="dapo_output",
        per_device_train_batch_size=4,
        num_generations=8,
        max_completion_length=1024,
        learning_rate=1e-6,
        beta=0.01,  # Lower beta for DAPO
        temperature=0.7,
        max_steps=200,
        logging_steps=5,
        save_steps=50,
        optim="adamw_8bit",
        # DAPO specific
        epsilon=0.2,
        epsilon_high=0.28,
        clip_ratio_c=3.0,  # DAPO clip ratio
        report_to="none",
    ),
)

trainer.train()

# Save
model.save_pretrained("dapo_lora")
tokenizer.save_pretrained("dapo_lora")

FastLanguageModel.save_pretrained_merged(
    model, tokenizer, "dapo_merged", save_method="merged_16bit"
)