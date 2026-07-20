from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
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

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")

# Format dataset (Alpaca -> ShareGPT)
def format_alpaca(example):
    return {
        "conversations": [
            {"role": "user", "content": example["instruction"] + "\n" + example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
    }

dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

# Train SFT
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="sft_output",
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=100,
        max_seq_length=2048,
        logging_steps=10,
        save_steps=50,
        packing=True,
    ),
)

trainer.train()

# Save
model.save_pretrained("sft_lora")
model.save_pretrained_merged("sft_merged", tokenizer)
model.save_pretrained_gguf("sft_gguf", tokenizer)