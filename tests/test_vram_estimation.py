import sys
sys.path.insert(0, '/root/Desktop/llm/.opencode/skills/llm-training-comprehensive')
from scripts.estimate_vram import estimate_vram

def test_vram_estimation_llama_8b():
    # 8B model, fp16/bf16, seq len 2048, batch 2, LoRA SFT
    result = estimate_vram(model_params_b=8.0, seq_len=2048, batch_size=2, quantize="4bit")
    assert "estimated_vram_gb" in result
    assert result["estimated_vram_gb"] < 16.0  # Should fit on a T4 (16GB)
    assert result["feasible_t4"] is True
