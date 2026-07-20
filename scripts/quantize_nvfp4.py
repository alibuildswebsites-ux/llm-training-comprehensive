#!/usr/bin/env python3
"""
NVFP4 Quantization Script for Blackwell GPUs (RTX 5090, B200)
Converts merged HF model to NVFP4 format using TensorRT-LLM.
"""
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to NVFP4 format")
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--output", required=True, help="Output directory for NVFP4 model")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--calib-dataset", default="wikitext", help="Calibration dataset")
    parser.add_argument("--max-samples", type=int, default=128, help="Max calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048, help="Calibration sequence length")
    args = parser.parse_args()

    # Check if tensorrt-llm is available
    try:
        import tensorrt_llm
    except ImportError:
        print("ERROR: tensorrt-llm not installed. Use NVIDIA container:")
        print("  docker run --gpus all -it nvcr.io/nvidia/tensorrt-llm:latest")
        sys.exit(1)

    # Build command
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", args.model,
        "--output_dir", args.output,
        "--dtype", "nvfp4",
        "--tp_size", str(args.tp_size),
        "--pp_size", str(args.pp_size),
        "--calib_data", args.calib_dataset,
        "--max_calibration_samples", str(args.max_samples),
        "--max_seq_len", str(args.seq_len),
        "--quant_algo", "fp4",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()