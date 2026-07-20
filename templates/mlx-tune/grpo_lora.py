from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
)

# Load dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1000]")

# Reward functions
def reward_format(completions, **kwargs):
    """Reward for correct format"""
    rewards = []
    for c in completions:
        if "<|im_end|>" in c:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_length(completions, **kwargs):
    """Reward for reasonable length"""
    rewards = []
    for c in completions:
        length = len(c.split())
        if 50 <= length <= 500:
            rewards.append(1.0)
        else:
            rewards.append(0.5)
    return rewards

# Train GRPO
trainer = GRPOTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_functions=[reward_format, reward_length],
    args=GRPOConfig(
        output_dir="grpo_output",
        per_device_train_batch_size=2,
        learning_rate=1e-6,
        num_generations=4,
        max_completion_length=512,
        beta=0.04,
        temperature=0.7,
        max_steps=50,
        logging_steps=5,
        save_steps=25,
    ),
)

trainer.train()

# Save
model.save_pretrained("grpo_lora")  # LoRA adapters only
model.save_pretrained_merged("grpo_merged", tokenizer)  # Full merged model (16-bit)
model.save_pretrained_gguf("grpo_gguf", tokenizer)  # GGUF for llama.cpp