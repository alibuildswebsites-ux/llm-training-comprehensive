# LLM Training Comprehensive Skill — Installation Guide

A comprehensive, multi-CLI compatible skill for training and fine-tuning LLMs. Works with **opencode**, **Claude Code**, **Gemini CLI**, and other agents that support skill loading.

---

## Quick Start (All CLIs)

```bash
# Clone the skill
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git

# Or download and extract from GitHub releases
```

---

## CLI-Specific Installation

### 1. opencode

**Option A: Project-local (recommended)**
```bash
# Clone directly into your project's .opencode/skills/
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  .opencode/skills/llm-training-comprehensive
```

**Option B: Global (all projects)**
```bash
mkdir -p ~/.config/opencode/skills
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  ~/.config/opencode/skills/llm-training-comprehensive
```

**Restart opencode** — the skill will auto-load.

---

### 2. Claude Code

**Option A: Project-local**
```bash
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  .claude/skills/llm-training-comprehensive
```

**Option B: Global**
```bash
mkdir -p ~/.claude/skills
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  ~/.claude/skills/llm-training-comprehensive
```

**Restart Claude Code** — the skill appears in `/skill` menu.

---

### 3. Gemini CLI

**Option A: Project-local**
```bash
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  .gemini/skills/llm-training-comprehensive
```

**Option B: Global**
```bash
mkdir -p ~/.gemini/skills
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  ~/.gemini/skills/llm-training-comprehensive
```

---

### 4. Generic / Any CLI Agent

If your CLI agent loads skills from a directory, just clone the repo:

```bash
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  path/to/your/skills/llm-training-comprehensive
```

The skill works with any agent that:
- Reads `SKILL.md` from a skill directory
- Supports frontmatter (`name`, `description`, `tags`)
- Can reference files in `scripts/`, `templates/`, `config/`, `references/`

---

## Verify Installation

After installation and CLI restart, verify:

```bash
# opencode
opencode skill list | grep llm-training

# Claude Code
/skill list

# Gemini CLI
gemini skill list
```

You should see: `llm-training-comprehensive` with description about LLM training.

---

## Usage Examples

Once installed, ask your agent:

```
"Fine-tune Llama-3.1-8B on my medical dataset using QLoRA on free Kaggle T4"
```

```
"Export my trained model to GGUF Q4_K_M for Ollama"
```

```
"Debug OOM error when training 13B model on RTX 3090"
```

```
"Set up DPO alignment after SFT — what's the best beta and learning rate?"
```

The agent will:
1. Guide you through **Spec → Plan → Prepare → Execute → Validate → Export**
2. Generate framework-specific configs (Unsloth, Axolotl, TRL, etc.)
3. Run dataset prep, VRAM estimation, training, and export scripts
4. Provide diagnostic help for OOM, NaN, convergence issues

---

## Skill Structure

```
llm-training-comprehensive/
├── SKILL.md                    # Main skill entry point (2,250+ lines)
├── config/                     # YAML configs
│   ├── model-templates.yaml    # 10 model families
│   ├── hardware-profiles.yaml  # GPU VRAM specs
│   ├── framework-versions.yaml # Pinned versions
│   ├── training-methods.yaml   # Method→framework mapping
│   └── export-formats.yaml     # GGUF/AWQ/GPTQ/EXL2/ONNX/TensorRT-LLM
├── templates/                  # Ready-to-run templates
│   ├── unsloth/               # SFT/DPO/GRPO/Full FT
│   ├── axolotl/               # YAML configs
│   ├── llama-factory/         # CLI configs
│   ├── trl/                   # SFT/DPO/GRPO trainers
│   ├── torchtune/             # PyTorch-native configs
│   └── raw-transformers/      # Maximum control
├── scripts/                    # Executable utilities
│   ├── prepare_dataset.py      # Alpaca→ShareGPT converter
│   ├── estimate_vram.py        # VRAM calculator
│   ├── launch_training.py      # Multi-platform launcher
│   ├── merge_lora.py           # Adapter merger
│   ├── convert_to_gguf.py      # llama.cpp conversion
│   ├── convert_to_awq.py       # AWQ quantization
│   ├── convert_to_gptq.py      # GPTQ quantization
│   ├── eval_model.py           # Perplexity/GSM8K/MMLU
│   ├── push_to_hub.py          # HF Hub upload
│   ├── colab_setup.py          # Colab auto-setup
│   ├── kaggle_setup.py         # Kaggle auto-setup
│   ├── diagnose_crash.py       # OOM/NaN/hang analysis
│   └── monitor_training.py     # Log monitoring + early stop
├── references/                 # Deep-dive guides
│   ├── method-comparison.md    # SFT vs DPO vs KTO vs GRPO...
│   ├── model-specifics.md      # Per-model quirks
│   ├── hardware-limits.md      # VRAM tables
│   ├── optimization-guide.md   # FlashAttn, GaLore, packing...
│   ├── troubleshooting.md      # Diagnostic playbook
│   └── colab-kaggle-guide.md   # Free-tier specifics
└── tests/                      # Validation tests
```

---

## Requirements

| Component | Minimum Version |
|-----------|-----------------|
| Python    | 3.9+            |
| PyTorch   | 2.1+ (CUDA 11.8+) |
| transformers | 4.46+         |
| peft      | 0.13+           |
| trl       | 0.12+           |
| unsloth   | 2024.11+        |
| bitsandbytes | 0.44+         |

Install core deps:
```bash
pip install unsloth transformers datasets peft accelerate trl bitsandbytes wandb
# For Flash Attention (A100/RTX 3090+/H100 only):
pip install flash-attn --no-build-isolation
```

---

## Free Tier Quickstart

### Kaggle (30 hrs/week, T4/P100)
```bash
# In Kaggle notebook:
exec(open('scripts/kaggle_setup.py').read())
python scripts/estimate_vram.py --params-b 8 --seq-len 4096 --batch-size 2 --quantize 4bit
# → Then follow agent guidance
```

### Google Colab (T4, 12hr sessions)
```bash
# In Colab notebook:
exec(open('scripts/colab_setup.py').read())
# Run keep-alive cell in background
```

---

## Contributing

PRs welcome! Areas needing help:
- More framework templates (LLaMA-Factory DPO, TorchTune DPO, etc.)
- Additional model families in `config/model-templates.yaml`
- More export formats (EXL2, ONNX, TensorRT-LLM)
- Test coverage for scripts

---

## License

MIT License — Use freely in commercial and personal projects.

---

## Links

- **Repository**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive
- **Issues**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive/issues
- **Discussions**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive/discussions