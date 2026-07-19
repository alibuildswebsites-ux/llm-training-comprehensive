#!/usr/bin/env python3
"""
Merge LoRA adapters into base model
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Base model path or HF ID")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
                       help="Output dtype")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                       help="Device to load model on")
    parser.add_argument("--save-tokenizer", action="store_true", help="Also save tokenizer")
    args = parser.parse_args()
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    print("Merging adapter...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output}")
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    
    if args.save_tokenizer:
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        tokenizer.save_pretrained(args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()