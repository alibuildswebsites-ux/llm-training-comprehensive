#!/usr/bin/env python3
"""
EXL2 Quantization Script for ExLlamaV2
Converts merged HF model to EXL2 format for fastest inference.
"""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to EXL2 format")
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--output", required=True, help="Output directory for EXL2 model")
    parser.add_argument("--bpw", type=float, default=4.0, choices=[2.5, 3.0, 4.0, 5.0, 6.0], help="Bits per weight")
    parser.add_argument("--calib-dataset", default="wikitext", help="Calibration dataset")
    parser.add_argument("--max-samples", type=int, default=128, help="Max calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048, help="Calibration sequence length")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "exllamav2.convert",
        "-i", args.model,
        "-o", args.output,
        "-b", str(args.bpw),
        "--calib_dataset", args.calib_dataset,
        "--max_samples", str(args.max_samples),
        "--seq_len", str(args.seq_len),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()