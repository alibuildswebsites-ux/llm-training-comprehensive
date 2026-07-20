#!/usr/bin/env python3
"""
Multimodal Training Script
Supports Qwen2-VL, Llama-3.2-Vision, Pixtral, Qwen2-Audio
"""
import argparse
from datasets import load_dataset
from transformers import TrainingArguments
import torch

def main():
    parser = argparse.ArgumentParser(description="Multimodal SFT/DPO training")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--method", choices=["sft", "dpo"], default="sft", help="Training method")
    parser.add_argument("--framework", choices=["unsloth", "trl", "axolotl"], default="unsloth", help="Framework")
    parser.add_argument("--output", default="multimodal_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--freeze-vision", action="store_true", default=True, help="Freeze vision encoder")
    parser.add_argument("--freeze-audio", action="store_true", default=True, help="Freeze audio encoder")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit", help="Quantization")
    parser.add_argument("--report-to", default="wandb", help="Report to (wandb, tensorboard, none)")
    args = parser.parse_args()

    if args.framework == "unsloth":
        train_unsloth(args)
    elif args.framework == "trl":
        train_trl(args)
    else:
        print(f"Framework {args.framework} not yet implemented in this script")
        print("Use axolotl config directly: axolotl train config.yaml")

def train_unsloth(args):
    from unsloth import FastVisionModel
    from trl import SFTTrainer, DPOTrainer
    
    # Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=args.quantize == "4bit",
        load_in_8bit=args.quantize == "8bit",
        use_gradient_checkpointing="unsloth",
    )

    # Add LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not args.freeze_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # Load dataset
    dataset = load_dataset(args.dataset, split="train")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to=args.report_to if args.report_to != "none" else "none",
        remove_unused_columns=False,
    )

    # Trainer
    if args.method == "sft":
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=FastVisionModel.data_collator,
            args=training_args,
        )
    else:  # dpo
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=FastVisionModel.data_collator,
            args=training_args,
        )

    trainer.train()

    # Save
    model.save_pretrained(f"{args.output}_lora")
    tokenizer.save_pretrained(f"{args.output}_lora")

    # Merge
    FastVisionModel.save_pretrained_merged(
        model, tokenizer, args.output, save_method="merged_16bit"
    )

def train_trl(args):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Freeze vision
    if args.freeze_vision:
        for param in model.visual.parameters():
            param.requires_grad = False

    # Dataset
    dataset = load_dataset(args.dataset, split="train")

    # Format
    def format_example(example):
        conversations = example["conversations"]
        images = example["images"]
        text = processor.apply_chat_template(conversations, tokenize=False)
        return {"text": text, "images": images}

    dataset = dataset.map(format_example)

    # Collator
    def collate_fn(examples):
        texts = [e["text"] for e in examples]
        images = [e["images"] for e in examples]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        return inputs

    # Training
    training_args = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        report_to=args.report_to if args.report_to != "none" else "none",
        gradient_checkpointing=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(f"{args.output}_lora")

if __name__ == "__main__":
    main()