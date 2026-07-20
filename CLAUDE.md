# llm-training-comprehensive — Skill for Claude Code

This skill provides comprehensive LLM training/fine-tuning capabilities for any CLI agent. Here's how to use it in **Claude Code**.

---

## Quick Start in Claude Code

### Option 1: Use as Reference (Recommended)

Just tell Claude Code what you want:

```
"I want to fine-tune Llama-3.1-8B on my medical dataset using QLoRA on a T4 GPU. 
Can you help me plan and execute this?"
```

Claude Code will use this skill's knowledge to:
- Select the right framework (Unsloth for T4)
- Configure QLoRA with optimal settings
- Set up dataset preparation
- Handle checkpointing for free tier
- Export to GGUF for deployment

---

### Option 2: Load Skill Context Explicitly

In your Claude Code session:

```
@/path/to/llm-training-comprehensive/SKILL.md
```

Or if cloned locally:
```
@./llm-training-comprehensive/SKILL.md
```

Then ask:
```
"Using this skill, help me set up DPO alignment for my model after SFT."
```

---

## Key Files for Reference

| File | Purpose |
|------|---------|
| `SKILL.md` | Full 2,250-line guide — all methods, models, platforms |
| `config/model-templates.yaml` | 10 model families with target_modules, chat templates |
| `config/hardware-profiles.yaml` | GPU VRAM specs (T4 → H100) |
| `config/training-methods.yaml` | SFT/DPO/KTO/ORPO/PPO/GRPO configs per framework |
| `config/export-formats.yaml` | GGUF/AWQ/GPTQ conversion commands |
| `scripts/estimate_vram.py` | Check if config fits in VRAM |
| `scripts/prepare_dataset.py` | Alpaca→ShareGPT converter |
| `templates/unsloth/` | Ready-to-run Python scripts |
| `templates/axolotl/` | YAML configs for axolotl train |

---

## Common Workflows

### SFT (Supervised Fine-Tuning)
```
"Fine-tune Qwen2.5-7B on my instruction dataset using QLoRA"
```

### DPO Alignment
```
"Align my SFT model using DPO with preference pairs"
```

### Free Tier (Colab/Kaggle)
```
"Train on Kaggle T4 with 30h/week quota — handle session persistence"
```

### Export for Deployment
```
"Export my trained model to GGUF Q4_K_M for Ollama"
```

---

## Multi-Framework Support

| Framework | Best For | Templates |
|-----------|----------|-----------|
| **Unsloth** | Free tiers, single GPU, speed | `templates/unsloth/` |
| **Axolotl** | Multi-GPU, reproducibility | `templates/axolotl/` |
| **LLaMA-Factory** | Beginners, WebUI | `templates/llama-factory/` |
| **TRL** | Research, custom RL | `templates/trl/` |
| **TorchTune** | PyTorch purity | `templates/torchtune/` |
| **Raw Transformers** | Maximum control | `templates/raw-transformers/` |

---

## Model Families Supported

Llama 2/3/3.1/3.2, Qwen 1.5/2/2.5/3, Mistral/Mixtral, Gemma 1/2/3, Phi 2/3/3.5, DeepSeek V2/V3, Yi, Baichuan, InternLM, StableLM

---

## Installation (Optional - for local scripts)

```bash
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive
cd llm-training-comprehensive
pip install -e .
```

Then use scripts directly:
```bash
python scripts/estimate_vram.py --params-b 8 --seq-len 4096 --batch-size 2 --quantize 4bit
python scripts/prepare_dataset.py --input data/alpaca.json --output data/sharegpt.json --format alpaca
```

---

## Resources

- **Full Guide**: `SKILL.md` (2,250 lines)
- **Installation Guide**: `INSTALL.md`
- **GitHub**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive
- **Issues**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive/issues