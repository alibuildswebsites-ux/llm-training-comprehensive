#!/usr/bin/env python3
"""
FP8 Quantization Script for Hopper/Blackwell GPUs
Converts merged HF model to FP8 (E4M3 or E5M2) format.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def quantize_to_fp8(model_path, output_path, dtype="fp8_e4m3", calib_dataset="wikitext", max_samples=128):
    """
    Quantize model to FP8.
    
    Args:
        model_path: Path to merged HF model
        output_path: Output directory
        dtype: "fp8_e4m3" (default) or "fp8_e5m2"
        calib_dataset: Calibration dataset
        max_samples: Number of calibration samples
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model with init_empty_weights for memory efficiency
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    
    # Load weights
    model = load_checkpoint_and_dispatch(model, model_path, device_map="auto")
    
    # FP8 quantization
    if dtype == "fp8_e4m3":
        torch_dtype = torch.float8_e4m3fn
    elif dtype == "fp8_e5m2":
        torch_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    
    # Quantize linear layers
    from transformers.quantizers import quantize
    from transformers.utils.quantization_config import FP8QuantizationConfig
    
    quant_config = FP8QuantizationConfig(
        dtype=torch_dtype,
        calibration_dataset=calib_dataset,
        calibration_samples=max_samples,
    )
    
    print(f"Quantizing to {dtype}...")
    model = quantize(model, quant_config)
    
    # Save
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("FP8 quantization complete!")

def main():
    parser = argparse.ArgumentParser(description="Quantize HF model to FP8")
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dtype", default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2"], help="FP8 format")
    parser.add_argument("--calib-dataset", default="wikitext", help="Calibration dataset")
    parser.add_argument("--max-samples", type=int, default=128, help="Max calibration samples")
    args = parser.parse_args()
    
    quantize_to_fp8(args.model, args.output, args.dtype, args.calib_dataset, args.max_samples)

if __name__ == "__main__":
    main()