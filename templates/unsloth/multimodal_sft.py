from unsloth import FastVisionModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load multimodal model (Qwen2-VL, Llama-3.2-Vision, Pixtral)
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Add LoRA to vision + language
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Load multimodal dataset
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train[:1000]")

# Format dataset for vision-language
def format_vlm(example):
    conversations = example["conversations"]
    images = example["images"]
    return {"conversations": conversations, "images": images}

dataset = dataset.map(format_vlm)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=FastVisionModel.data_collator,
    args=TrainingArguments(
        output_dir="vlm_sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
    ),
)

trainer.train()

# Save
model.save_pretrained("vlm_lora")
tokenizer.save_pretrained("vlm_lora")

# Merge and save
FastVisionModel.save_pretrained_merged(
    model, tokenizer, "vlm_merged", save_method="merged_16bit"
)