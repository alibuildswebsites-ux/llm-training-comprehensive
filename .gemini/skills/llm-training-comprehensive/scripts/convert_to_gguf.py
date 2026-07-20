#!/usr/bin/env python3
"""
Convert HF model to GGUF with quantization using llama.cpp
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF with quantization")
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--output", required=True, help="Output directory for GGUF files")
    parser.add_argument("--llama-cpp", default="llama.cpp", help="Path to llama.cpp repo")
    parser.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"], 
                       help="Quantization levels")
    parser.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    parser.add_argument("--skip-conversion", action="store_true", help="Skip HF->GGUF, only quantize")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llama_cpp_dir = Path(args.llama_cpp)
    if not llama_cpp_dir.exists():
        print(f"Cloning llama.cpp to {llama_cpp_dir}")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)], check=True)
    
    # Build llama.cpp
    print("Building llama.cpp...")
    subprocess.run(["make", "-C", str(llama_cpp_dir), "clean"], check=False)
    subprocess.run(["make", "-C", str(llama_cpp_dir)], check=True)
    
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "llama-quantize"
    
    # Step 1: Convert HF to GGUF F16
    f16_path = output_dir / "model-f16.gguf"
    
    if not args.skip_conversion:
        print(f"Converting {model_path} to GGUF F16...")
        subprocess.run([
            sys.executable, str(convert_script),
            str(model_path),
            "--outfile", str(f16_path),
            "--outtype", args.outtype,
        ], check=True)
    else:
        # Find existing f16
        existing = list(output_dir.glob("*f16*.gguf"))
        if existing:
            f16_path = existing[0]
            print(f"Using existing: {f16_path}")
        else:
            print("No F16 model found and --skip-conversion set!")
            return 1
    
    # Step 2: Quantize
    for quant in args.quants:
        out_path = output_dir / f"model-{quant}.gguf"
        print(f"Quantizing to {quant}...")
        subprocess.run([
            str(quantize_bin),
            str(f16_path),
            str(out_path),
            quant,
        ], check=True)
        print(f"  Saved: {out_path}")
    
    # Cleanup
    if f16_path.exists() and "Q4_K_M" in args.quants:
        # Keep F16 only if requested
        if "F16" not in args.quants and "f16" not in args.quants:
            f16_path.unlink()
    
    print("Done!")
    print(f"Models saved to: {output_dir}")

if __name__ == "__main__":
    main()