# Method Comparison Reference

## Quick Decision Matrix

```
What is your goal?
│
├─→ Teach model new knowledge / domain / style
│   └─→ SFT (Supervised Fine-Tuning)
│
├─→ Align model with human preferences
│   │
│   ├─→ Have chosen/rejected pairs
│   │   ├─→ Want simplest, no ref model
│   │   │   └─→ ORPO (single-stage)
│   │   ├─→ Want proven, standard approach
│   │   │   └─→ DPO
│   │   ├─→ Have unpaired feedback (thumbs up/down)
│   │   │   └─→ KTO
│   │   ├─→ ORPO unstable, want contrastive
│   │   │   └─→ CPO
│   │   └─→ Want length normalization
│   │       └─→ SimPO
│   │
│   └─→ Have trained reward model
│       ├─→ Need fine-grained control
│       │   └─→ PPO
│       └─→ Reasoning tasks (math/code)
│           └─→ GRPO
│
└─→ Need reward model for PPO
    └─→ Reward Modeling
```

---

## Detailed Comparison

| Aspect | SFT | DPO | KTO | ORPO | CPO | SimPO | PPO | GRPO | Reward Model |
|--------|-----|-----|-----|------|-----|-------|-----|------|--------------|
| **Data Format** | (Q, A) | (Q, A+, A-) | (Q, A, label) | (Q, A+, A-) | (Q, A+, A-) | (Q, A+, A-) | (Q, A, r) | (Q, A×N, r×N) | (Q, A+, A-) |
| **Ref Model** | No | Yes | Yes | No | No | No | Yes | No | No |
| **Reward Model** | No | No | No | No | No | No | Yes | No | Output |
| **Stages** | 1 | 2 (SFT→DPO) | 2 (SFT→KTO) | 1 | 2 (SFT→CPO) | 2 (SFT→SimPO) | 3+ (SFT→RM→PPO) | 2 (SFT→GRPO) | 1 |
| **Complexity** | Low | Medium | Medium | Low | Medium | Medium | High | Medium | Medium |
| **VRAM (8B)** | 8GB | 9GB | 9GB | 8GB | 8GB | 8GB | 12GB+ | 9GB | 8GB |
| **Training Time** | Fast | Medium | Medium | Fast | Medium | Medium | Slow | Medium | Medium |
| **Data Difficulty** | Easy | Hard (pairs) | Medium (labels) | Hard (pairs) | Hard (pairs) | Hard (pairs) | Hard (rewards) | Medium (rewards) | Hard (pairs) |
| **Best For** | Domain, instruct | Standard align | Noisy/unpaired | Simple pipeline | Stable contrast | Length bias | Complex RL | Reasoning | PPO prep |

---

## When to Use Each Method

### SFT - Always First
```
✓ New domain (medical, legal, code)
✓ New language
✓ Instruction following
✓ Style transfer
✓ Format adherence (JSON, XML)
✓ Foundation for ALL alignment
```

### DPO - Standard Alignment
```
✓ After SFT, have preference pairs
✓ Clear chosen > rejected
✓ Want proven, well-tested method
✓ Benchmark against literature
```

### KTO - Practical Feedback
```
✓ User feedback: thumbs up/down
✓ Noisy labels (some wrong)
✓ Can't create pairs easily
✓ Binary classification easier
```

### ORPO - Simplest Alignment
```
✓ Want single-stage training
✓ Don't have/want reference model
✓ Quick experiment
✓ Limited compute
```

### CPO - When ORPO Fails
```
✓ ORPO unstable or poor results
✓ Want contrastive learning benefits
✓ Preference data available
```

### SimPO - Length Normalization
```
✓ DPO overfits to verbose responses
✓ Chosen/rejected differ mainly in length
✓ Want reference-free DPO variant
```

### PPO - Maximum Control
```
✓ Trained reward model available
✓ Complex multi-objective alignment
✓ Need KL control per token
✓ Research / novel algorithms
✓ Have compute for 3-stage pipeline
```

### GRPO - Reasoning Tasks
```
✓ Math, code, logic reasoning
✓ Can define verifiable rewards
✓ More stable than PPO
✓ No value network needed
✓ DeepSeek-R1 style training
```

### Reward Modeling - PPO Prerequisite
```
✓ Need reward model for PPO
✓ Large preference dataset
✓ Want to evaluate model outputs
✓ Ensemble for better reward
```

---

## Pipeline Recommendations

### Minimal (Free Tier, 1 GPU)
```
SFT (QLoRA) → DPO/ORPO → GGUF
Time: 4-8 hours on T4
VRAM: 8GB max
```

### Standard (Cloud, 1 GPU)
```
SFT (LoRA) → DPO → Eval → GGUF + AWQ
Time: 2-4 hours on A100
VRAM: 16-24GB
```

### Quality (Multi-GPU, 2-4 GPUs)
```
SFT (Full/LoRA) → DPO/KTO → PPO/GRPO → Full Eval → Multi-format export
Time: 8-24 hours on A100x4
VRAM: 40-80GB per GPU
```

### Reasoning Specialist
```
SFT (Code/Math) → GRPO (with verifiable rewards) → Eval on GSM8K/MATH → Export
Time: 6-12 hours on A100x4
VRAM: 40-80GB
```

---

## Hyperparameter Quick Reference

### SFT
```yaml
learning_rate: 2e-4 (LoRA) / 2e-5 (Full)
num_epochs: 3 (LoRA) / 1-2 (Full)
max_seq_length: 4096-8192
packing: true (single-turn) / false (multi-turn)
lora_r: 16-32
lora_alpha: 32-64
```

### DPO
```yaml
learning_rate: 5e-7
num_epochs: 1-3
beta: 0.1 (0.05-0.5)
max_length: 1024
max_prompt_length: 512
loss_type: sigmoid
```

### KTO
```yaml
learning_rate: 5e-7
num_epochs: 1-3
beta: 0.1
desirable_weight: 1.0
undesirable_weight: 1.0 (or adjust for imbalance)
```

### ORPO
```yaml
learning_rate: 1e-5
num_epochs: 1-3
beta: 0.1
max_length: 1024
```

### PPO
```yaml
learning_rate: 1e-6
ppo_epochs: 4
mini_batch_size: 16
batch_size: 64
kl_penalty: kl
init_kl_coef: 0.2
target_kl: 0.1
```

### GRPO
```yaml
learning_rate: 1e-6
num_generations: 8
max_completion_length: 512
beta: 0.04
temperature: 0.7
```

### Reward Modeling
```yaml
learning_rate: 1e-5
num_epochs: 1-2
max_length: 1024
```

---

## Data Requirements

| Method | Min Samples | Recommended | Format Quality |
|--------|-------------|-------------|----------------|
| SFT | 1,000 | 5,000-50,000 | High |
| DPO | 500 pairs | 2,000-10,000 pairs | Very High |
| KTO | 1,000 | 5,000+ | Medium (noise OK) |
| ORPO | 500 pairs | 2,000+ pairs | High |
| CPO | 500 pairs | 2,000+ pairs | High |
| SimPO | 500 pairs | 2,000+ pairs | High |
| PPO | 10,000 | 50,000+ | Medium (RL explores) |
| GRPO | 1,000 | 5,000+ | Medium (needs rewards) |
| Reward Model | 1,000 pairs | 10,000+ pairs | Very High |

---

## Common Pitfalls

| Method | Pitfall | Solution |
|--------|---------|----------|
| SFT | Overfitting on small data | More data, fewer epochs, higher dropout |
| DPO | Reference model mismatch | Use exact SFT checkpoint |
| DPO | Pair quality low | Filter pairs, human review |
| KTO | Imbalanced labels | Adjust weights, collect more |
| ORPO | Unstable loss | Lower LR, check data |
| PPO | Reward hacking | KL penalty, reward clipping |
| GRPO | Reward gaming | Multiple reward functions |
| All | Wrong chat template | Verify with tokenizer.apply_chat_template |

---

## Evaluation Strategy

### After SFT
- Perplexity on held-out data
- Instruction following (IFEval, MT-Bench)
- Domain-specific benchmarks

### After Alignment (DPO/KTO/ORPO/etc.)
- Win rate vs SFT (GPT-4 judge or human)
- Safety benchmarks (TruthfulQA, Toxigen)
- Style/format adherence
- Regression check on SFT capabilities

### After PPO/GRPO
- Task-specific metrics (accuracy on GSM8K, HumanEval)
- Reward model scores
- KL divergence from SFT
- Diversity metrics

### Export Validation
- GGUF: Test in Ollama/LM Studio
- AWQ/GPTQ: Test in vLLM/TGI
- Compare quantized vs original