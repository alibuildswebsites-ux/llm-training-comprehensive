# llm-training-comprehensive — Skill for Gemini CLI

Comprehensive LLM training/fine-tuning skill compatible with Gemini CLI.

---

## Quick Start

### Option 1: Context Injection
```
@/path/to/llm-training-comprehensive/SKILL.md

Help me fine-tune Llama-3.1-8B on my dataset using QLoRA.
```

### Option 2: Direct Questions
```
I need to train a model on a single T4 GPU. What's the best approach?
```
Gemini will use this skill's knowledge automatically if the skill is in your skill path.

---

## Installation

### Project-local
```bash
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  .gemini/skills/llm-training-comprehensive
```

### Global
```bash
mkdir -p ~/.gemini/skills
git clone https://github.com/alibuildswebsites-ux/llm-training-comprehensive.git \
  ~/.gemini/skills/llm-training-comprehensive
```

---

## Key Files for Quick Reference

| File | Use Case |
|------|----------|
| `SKILL.md` | Complete methodology |
| `config/model-templates.yaml` | Model-specific settings |
| `scripts/estimate_vram.py` | VRAM feasibility check |
| `templates/unsloth/sft_lora.py` | Fastest SFT template |

---

## Example Prompts

```
"Set up QLoRA for Llama-3.1-8B on Kaggle T4 with 4096 context"
```

```
"Debug OOM when training 13B on RTX 3090"
```

```
"Export trained model to GGUF Q4_K_M for Ollama"
```

---

## Repository
https://github.com/alibuildswebsites-ux/llm-training-comprehensive