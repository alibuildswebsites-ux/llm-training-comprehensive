def estimate_vram(model_params_b, seq_len, batch_size, quantize="none"):
    # Base model parameter memory
    if quantize == "4bit":
        bytes_per_param = 0.5
    elif quantize == "8bit":
        bytes_per_param = 1.0
    else:
        bytes_per_param = 2.0 # 16-bit precision

    model_memory = model_params_b * bytes_per_param
    
    # Activation memory estimate (highly non-linear, but approximate rule of thumb)
    # 1B params at 2048 seq, batch 1 is around ~0.3 GB VRAM
    activation_memory = (model_params_b * 0.1) * (seq_len / 2048.0) * batch_size
    
    # Optimizer and gradient states (LoRA approx)
    optimizer_memory = 1.5 if quantize != "none" else (model_params_b * 4.0)
    
    total_memory = model_memory + activation_memory + optimizer_memory
    
    # Overhead/CUDA context
    total_memory += 1.5
    
    return {
        "estimated_vram_gb": round(total_memory, 2),
        "feasible_t4": total_memory < 15.0,
        "feasible_rtx3090": total_memory < 23.0,
        "feasible_a100": total_memory < 78.0,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-b", type=float, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quantize", default="4bit")
    args = parser.parse_args()
    
    res = estimate_vram(args.params_b, args.seq_len, args.batch_size, args.quantize)
    print(f"Estimated VRAM: {res['estimated_vram_gb']} GB")
