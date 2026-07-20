from unsloth import FastVisionModel
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load multimodal model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Add LoRA
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
)

# Load multimodal DPO dataset (chosen/rejected with images)
dataset = load_dataset("trl-lib/ultrafeedback-binarized-images", split="train[:1000]")

# Train DPO
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Use model as reference
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=FastVisionModel.data_collator,
    args=TrainingArguments(
        output_dir="vlm_dpo_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=50,
        learning_rate=5e-7,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
    ),
)

trainer.train()

# Save
model.save_pretrained("vlm_dpo_lora")
tokenizer.save_pretrained("vlm_dpo_lora")

# Merge
FastVisionModel.save_pretrained_merged(
    model, tokenizer, "vlm_dpo_merged", save_method="merged_16bit"
)