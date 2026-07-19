#!/usr/bin/env python3
"""
Convert HF model to GPTQ format
"""

import argparse
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Quantize model to GPTQ")
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--output", required=True, help="Output path for GPTQ model")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4, 8], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--calib-data", default="wikitext", help="Calibration dataset")
    parser.add_argument("--calib-samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
    parser.add_argument("--desc-act", action="store_true", help="Use desc_act quantization")
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print(f"Quantizing to {args.bits}-bit GPTQ (group_size={args.group_size})...")
    model.quantize(
        tokenizer,
        quant_config={
            "bits": args.bits,
            "group_size": args.group_size,
            "desc_act": args.desc_act,
        },
        calib_data=args.calib_data,
        calib_samples=args.calib_samples,
        calib_seqlen=args.seqlen,
    )
    
    print(f"Saving to: {args.output}")
    model.save_quantized(args.output, use_safetensors=True)
    tokenizer.save_pretrained(args.output)
    
    print("GPTQ quantization complete!")

if __name__ == "__main__":
    main()