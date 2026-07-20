import os
import yaml

def test_training_methods_config():
    """Test the new training-methods.yaml config with 14 methods"""
    path = "config/training-methods.yaml"
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
        assert data is not None
        # Check for 14 methods
        expected_methods = ["sft", "dpo", "kto", "orpo", "ppo", "grpo", "dapo", 
                           "cpo", "simpo", "reward-modeling", "continued-pretraining",
                           "multimodal-sft", "multimodal-dpo", "mlx-sft", "mlx-dpo", "mlx-grpo"]
        for method in expected_methods:
            assert method in data, f"Method {method} not found in training-methods.yaml"

def test_export_formats_config():
    """Test the new export-formats.yaml config with 10 formats"""
    path = "config/export-formats.yaml"
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
        assert data is not None
        # Check for 10 export formats
        expected_formats = ["gguf", "awq", "gptq", "exl2", "exl3", "fp8", "nvfp4", "mxfp4", "hf-safetensors", "onnx", "tensorrt-llm"]
        for fmt in expected_formats:
            assert fmt in data, f"Format {fmt} not found in export-formats.yaml"

def test_hardware_profiles_2026():
    """Test hardware-profiles.yaml has 2026 GPUs"""
    path = "config/hardware-profiles.yaml"
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
        assert data is not None
        # Check for new 2026 GPUs
        new_gpus = ["RTX5090", "H200_141GB", "B200_192GB", "MI325X_256GB", "Apple_M4_Max", "Apple_M4_Ultra"]
        for gpu in new_gpus:
            assert gpu in data, f"GPU {gpu} not found in hardware-profiles.yaml"

def test_framework_versions_2026():
    """Test framework-versions.yaml has 2026 versions"""
    path = "config/framework-versions.yaml"
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
        assert data is not None
        # Check for 2026 versions
        assert data.get("unsloth") == "2026.7.1", f"Unsloth version should be 2026.7.1, got {data.get('unsloth')}"
        assert data.get("trl") == "0.16.0", f"TRL version should be 0.16.0, got {data.get('trl')}"
        assert data.get("torchtune") == "0.5.0", f"TorchTune version should be 0.5.0, got {data.get('torchtune')}"
        assert data.get("unsloth-mlx") == "0.2.0", f"Unsloth-MLX version should be 0.2.0, got {data.get('unsloth-mlx')}"

def test_new_scripts_exist():
    """Test new 2026 scripts exist"""
    scripts = [
        "scripts/quantize_exl2.py",
        "scripts/quantize_fp8.py",
        "scripts/quantize_nvfp4.py",
        "scripts/train_multimodal.py"
    ]
    for script in scripts:
        assert os.path.exists(script), f"Script {script} does not exist"

def test_new_templates_exist():
    """Test new 2026 templates exist"""
    templates = [
        "templates/mlx-tune/sft_lora.py",
        "templates/mlx-tune/dpo_lora.py",
        "templates/mlx-tune/grpo_lora.py",
        "templates/unsloth/multimodal_sft.py",
        "templates/unsloth/multimodal_dpo.py",
        "templates/unsloth/grpo_lora.py",
        "templates/unsloth/dapo_lora.py",
        "templates/torchtune/dpo_lora.yaml",
        "templates/torchtune/grpo_lora.yaml",
        "templates/axolotl/multimodal_sft.yaml",
        "templates/axolotl/grpo_lora.yaml",
        "templates/llama-factory/multimodal_sft.yaml",
        "templates/trl/multimodal_sft.py"
    ]
    for template in templates:
        assert os.path.exists(template), f"Template {template} does not exist"