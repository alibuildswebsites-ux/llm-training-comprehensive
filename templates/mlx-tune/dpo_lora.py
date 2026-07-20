from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
)

# Load DPO dataset (chosen/rejected pairs)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:2000]")

# Train DPO
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=DPOConfig(
        output_dir="dpo_output",
        per_device_train_batch_size=2,
        learning_rate=5e-7,
        max_steps=100,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        logging_steps=10,
        save_steps=50,
    ),
)

trainer.train()

# Save
model.save_pretrained("dpo_lora")
model.save_pretrained_merged("dpo_merged", tokenizer)
model.save_pretrained_gguf("dpo_gguf", tokenizer)