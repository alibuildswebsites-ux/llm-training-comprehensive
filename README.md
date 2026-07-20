# llm-training-comprehensive

A comprehensive, multi-CLI compatible skill for training and fine-tuning LLMs on any hardware — from free Kaggle T4 to multi-GPU cloud clusters.

[![GitHub stars](https://img.shields.io/github/stars/alibuildswebsites-ux/llm-training-comprehensive?style=social)](https://github.com/alibuildswebsites-ux/llm-training-comprehensive)
[![GitHub license](https://img.shields.io/github/license/alibuildswebsites-ux/llm-training-comprehensive)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## 🎯 What This Does

This skill enables **any CLI agent** (opencode, Claude Code, Gemini CLI, etc.) to help you:

- **Train/fine-tune** any open-source LLM (Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, Yi, Baichuan, InternLM, StableLM)
- **Choose the right framework** — Unsloth, Axolotl, LLaMA-Factory, TRL, TorchTune, Raw Transformers
- **Pick the best method** — SFT, DPO, KTO, ORPO, PPO, GRPO, CPO, SimPO, Reward Modeling
- **Use any PEFT technique** — LoRA, QLoRA, DoRA, VeRA, LoHa, LoKr, OFT, AdaLoRA, IA³
- **Run on any platform** — Colab, Kaggle, local, RunPod, Lambda, Vast.ai, multi-GPU
- **Export to any format** — GGUF, AWQ, GPTQ, EXL2, ONNX, TensorRT-LLM, HF safetensors
- **Debug issues** — OOM, NaN losses, convergence, chat template mismatches

---

## ⚡ Quick Start

### Install (All CLIs)

```bash
# Project-local (recommended)
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  path/to/your/skills/llm-training-comprehensive
```

### CLI-Specific Global Install

| CLI | Command |
|-----|---------|
| **opencode** | `git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git ~/.config/opencode/skills/llm-training-comprehensive` |
| **Claude Code** | `git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git ~/.claude/skills/llm-training-comprehensive` |
| **Gemini CLI** | `git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git ~/.gemini/skills/llm-training-comprehensive` |

**Restart your CLI** — skill auto-loads.

---

## 🧠 What You Can Ask Your Agent

Once installed, just ask naturally:

```
"Fine-tune Llama-3.1-8B on my medical dataset using QLoRA on free Kaggle T4"
```

```
"Set up DPO alignment after SFT — what beta and learning rate should I use?"
```

```
"Debug OOM error when training 13B model on RTX 3090"
```

```
"Export my trained model to GGUF Q4_K_M for Ollama"
```

```
"Train on Colab with session persistence and auto-reconnect"
```

---

## 📁 Skill Structure

```
llm-training-comprehensive/
├── SKILL.md                    # 2,250+ line main guide
├── INSTALL.md                  # Complete installation guide
├── CLAUDE.md                   # Claude Code quick start
├── GEMINI.md                   # Gemini CLI quick start
├── config/                     # YAML configurations
│   ├── model-templates.yaml    # 10 model families
│   ├── hardware-profiles.yaml  # GPU VRAM specs
│   ├── framework-versions.yaml # Pinned versions
│   ├── training-methods.yaml   # Method→framework mappings
│   └── export-formats.yaml     # GGUF/AWQ/GPTQ/EXL2/ONNX/TensorRT-LLM
├── templates/                  # Ready-to-run configs (6 frameworks)
│   ├── unsloth/               # SFT/DPO/GRPO/Full FT
│   ├── axolotl/               # YAML configs
│   ├── llama-factory/         # CLI configs
│   ├── trl/                   # SFT/DPO/GRPO trainers
│   ├── torchtune/             # PyTorch-native configs
│   └── raw-transformers/      # Maximum control
├── scripts/                    # 14 executable utilities
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
├── references/                 # 6 deep-dive guides
│   ├── method-comparison.md    # SFT vs DPO vs KTO vs GRPO...
│   ├── model-specifics.md      # Per-model quirks
│   ├── hardware-limits.md      # VRAM tables
│   ├── optimization-guide.md   # FlashAttn, GaLore, packing...
│   ├── troubleshooting.md      # Diagnostic playbook
│   └── colab-kaggle-guide.md   # Free-tier specifics
└── tests/                      # 8 passing tests
```

---

## 🛠 Frameworks Supported

| Framework | Best For | Templates |
|-----------|----------|-----------|
| **Unsloth** | Free tiers, single GPU, speed | Python scripts |
| **Axolotl** | Multi-GPU, reproducibility | YAML only |
| **LLaMA-Factory** | Beginners, WebUI | YAML + CLI |
| **TRL** | Research, custom RL | Python scripts |
| **TorchTune** | PyTorch purity | YAML |
| **Raw Transformers** | Maximum control | Python scripts |

---

## 🤖 Model Families (10)

| Family | Variants | Target Modules |
|--------|----------|----------------|
| **Llama** | 2/3/3.1/3.2 (1B-405B) | q/k/v/o/gate/up/down_proj |
| **Qwen** | 1.5/2/2.5/3 (0.6B-235B) | q/k/v/o/gate/up/down_proj |
| **Mistral/Mixtral** | 7B, 8x7B, 8x22B | q/k/v/o/gate/up/down_proj |
| **Gemma** | 1/2/3 (1B-27B) | q/k/v/o/gate/up/down_proj |
| **Phi** | 2/3/3.5 (2.7B-14B) | qkv_proj, gate/up/down_proj |
| **DeepSeek** | V2/V3 (16B-671B) | q/k/v/o/gate/up/down_proj |
| **Yi** | 6B/9B/34B | q/k/v/o/gate/up/down_proj |
| **Baichuan** | 7B/13B | W_pack, gate/up/down_proj |
| **InternLM** | 7B/20B | qkv_proj, gate/up/down_proj |
| **StableLM** | 3B/7B | q/k/v/o/gate/up/down_proj |

---

## 📊 Methods Supported

| Method | Data | Ref Model | Best For |
|--------|------|-----------|----------|
| **SFT** | (instruction, response) | No | Domain adaptation, instruction following |
| **DPO** | (prompt, chosen, rejected) | Yes | Standard preference alignment |
| **KTO** | (prompt, response, label) | Yes | Unpaired/noisy feedback |
| **ORPO** | (prompt, chosen, rejected) | No | Single-stage, no ref model |
| **CPO** | (prompt, chosen, rejected) | No | Stable contrastive |
| **SimPO** | (prompt, chosen, rejected) | No | Length-normalized |
| **PPO** | (prompt, response, reward) | Yes | Complex RL, fine-grained control |
| **GRPO** | (prompt, responses, rewards) | No | Reasoning, math, code |
| **Reward Modeling** | (prompt, chosen, rejected) | No | PPO prerequisite |

---

## 💰 Free Tier Support

### Kaggle (Best Free Option)
- **30 hrs/week** GPU quota (T4 or P100)
- **No idle timeout** (unlike Colab)
- **Persistent storage** at `/kaggle/working`

### Google Colab
- **T4 16GB** GPU
- **12hr session**, 90min idle timeout
- **Auto-reconnect** via keep-alive cell

```bash
# In notebook:
exec(open('scripts/kaggle_setup.py').read())
python scripts/estimate_vram.py --params-b 8 --seq-len 4096 --batch-size 2 --quantize 4bit
```

---

## 🚀 Export Pipeline

```
Train → Merge LoRA → HF safetensors → GGUF (Ollama/LM Studio)
                              → AWQ (vLLM/TGI)
                              → GPTQ (vLLM/TGI)
                              → EXL2 (ExLlamaV2)
                              → ONNX/TensorRT-LLM (max inference speed)
```

---

## 🔧 Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.9+ |
| PyTorch | 2.1+ (CUDA 11.8+) |
| transformers | 4.46+ |
| peft | 0.13+ |
| trl | 0.12+ |
| unsloth | 2024.11+ |
| bitsandbytes | 0.44+ |

```bash
pip install unsloth transformers datasets peft accelerate trl bitsandbytes wandb
# Flash Attention (A100/RTX 3090+/H100 only):
pip install flash-attn --no-build-isolation
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Or run specific test
python -m pytest tests/test_vram_estimation.py -v
```

---

## 📚 Documentation

- **[SKILL.md](SKILL.md)** — Complete 2,250-line methodology guide
- **[INSTALL.md](INSTALL.md)** — Detailed installation & usage
- **[CLAUDE.md](CLAUDE.md)** — Claude Code quick reference
- **[GEMINI.md](GEMINI.md)** — Gemini CLI quick reference
- **[references/](references/)** — 6 deep-dive guides

---

## 🤝 Contributing

PRs welcome! Priority areas:
- More framework templates (LLaMA-Factory DPO, TorchTune DPO, etc.)
- Additional model families in `config/model-templates.yaml`
- More export formats (EXL2, ONNX, TensorRT-LLM)
- Test coverage for scripts

---

## 📄 License

MIT License — Free for commercial and personal use.

---

## 🔗 Links

- **Repository**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive
- **Issues**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive/issues
- **Discussions**: https://github.com/alibuildswebsites-ux/llm-training-comprehensive/discussions