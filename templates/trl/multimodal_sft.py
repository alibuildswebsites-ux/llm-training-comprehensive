# TRL Multimodal SFT Configuration
# Run: python templates/trl/multimodal_sft.py

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Load dataset
dataset = load_dataset("HuggingFaceM4/the_cauldron", "docvqa", split="train[:1000]")

def format_example(example):
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "images": [example["image"]]}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Freeze vision encoder
for param in model.visual.parameters():
    param.requires_grad = False

# Training args
training_args = SFTConfig(
    output_dir="trl_multimodal_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    report_to="wandb",
    run_name="trl-multimodal-sft",
    gradient_checkpointing=True,
    dataset_text_field="text",
    dataset_kwargs={"skip_prepare_dataset": True},
)

# Custom data collator for multimodal
def collate_fn(examples):
    texts = [e["text"] for e in examples]
    images = [e["images"] for e in examples]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return inputs

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=collate_fn,
)

trainer.train()

# Save
trainer.save_model("trl_multimodal_lora")
processor.save_pretrained("trl_multimodal_lora")