# Model-Specific Quirks & Configuration

This document captures per-model idiosyncrasies, special tokens, chat templates, and known issues.

---

## Llama Family

### Llama 2 (7B/13B/70B)
- **Architecture**: Standard decoder-only, GQA in 70B
- **Context**: 4096 tokens
- **Chat template**: 
  ```
  <s>[INST] <<SYS>>
  {system}
  <</SYS>>
  {user} [/INST] {assistant} </s>
  ```
- **Special tokens**: `<s>` (BOS), `</s>` (EOS), `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>`
- **Tokenizer**: `LlamaTokenizer` (sentencepiece)
- **Target modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Notes**: 
  - No RoPE scaling needed
  - 70B uses Grouped Query Attention (8 heads for Q, 1 for K/V)
  - Trust remote code: false

### Llama 3 (8B/70B)
- **Architecture**: Standard decoder, GQA in 70B
- **Context**: 8192 tokens
- **Chat template**: Uses special tokens `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`
  ```
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
  {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  {assistant}<|eot_id|>
  ```
- **Tokenizer**: `LlamaTokenizerFast` (tiktoken-based)
- **Special tokens**: 
  - `128000` = `<|begin_of_text|>`
  - `128001` = `<|end_of_text|>`
  - `128002-128255` = reserved
- **Target modules**: Same as Llama 2
- **Notes**:
  - Tokenizer auto-loads chat template
  - Pad token defaults to EOS
  - Vocabulary: 128256 tokens

### Llama 3.1 (8B/70B/405B)
- **Architecture**: Same as Llama 3
- **Context**: 128K tokens (uses RoPE scaling)
- **Chat template**: Identical to Llama 3
- **RoPE scaling**: `llama3` type with factor 8.0
- **Target modules**: Same
- **Notes**:
  - 405B requires multi-GPU (8x A100-80 or H100)
  - For >8K context, must enable RoPE scaling in config
  - Use `rope_scaling={"type": "llama3", "factor": 8.0}`

### Llama 3.2 (1B/3B/90B-Vision)
- **Architecture**: 1B/3B are small dense; 90B is vision model
- **Context**: 128K
- **Chat template**: Same as Llama 3.1
- **Vision variant**: Use `LlavaForCausalLM` for fine-tuning
- **Small models**: 1B/3B can full fine-tune on T4
- **Target modules**: Same pattern

---

## Qwen Family

### Qwen 1.5 (7B/14B/72B)
- **Architecture**: Standard decoder, GQA
- **Context**: 32K
- **Chat template**: Uses `{{ system }}`, `{{ user }}`, `{{ assistant }}` with special tokens
- **Tokenizer**: `Qwen2TokenizerFast` (BPE, tiktoken-based)
- **Trust remote code**: `true` (required for Qwen 1.5)
- **Target modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Notes**: 
  - Different vocab from Llama (151936 tokens)
  - System prompt handling differs

### Qwen 2 (7B/72B)
- **Architecture**: Improved Qwen 1.5, GQA
- **Context**: 32K (can extend with RoPE)
- **Chat template**: Same format as Qwen 1.5
- **Trust remote code**: `true` for some variants
- **Target modules**: Same as Qwen 1.5
- **Notes**: 
  - 7B: 28 layers, 70B: 80 layers
  - Better multilingual than 1.5

### Qwen 2.5 (3B/7B/14B/32B/72B)
- **Architecture**: Further improved
- **Context**: 128K
- **Chat template**: Same format
- **Trust remote code**: `true`
- **Target modules**: Same
- **Notes**:
  - 32B is quality/cost sweet spot
  - 3B excellent for edge
  - Best coding/math in series

### Qwen 3 (0.6B-235B)
- **Architecture**: Latest with "thinking mode"
- **Context**: 128K+
- **Chat template**: Supports hybrid thinking
  ```
  <|im_start|>system
  {system}<|im_end|>
  <|im_start|>user
  {user}<|im_end|>
  <|im_start|>assistant
  <|thinking|>{thinking}<|thinking_end|>
  {response}<|im_end|>
  ```
- **Trust remote code**: `true`
- **Target modules**: Same
- **Notes**:
  - Requires special handling for thinking tokens
  - 0.6B fits anywhere, 235B needs multi-node
  - Hybrid thinking can be enabled/disabled

---

## Mistral Family

### Mistral 7B (v0.1/v0.2/v0.3)
- **Architecture**: Sliding window attention (4096)
- **Context**: v0.1=8K (effective 4K), v0.2/v0.3=32K
- **Chat template**: 
  ```
  <s>[INST] {user} [/INST] {assistant} </s>
  ```
- **Tokenizer**: `MistralTokenizerFast` (BPE)
- **Target modules**: Standard 7 modules
- **Notes**:
  - Sliding window NOT applied during training (full attention)
  - v0.2/v0.3: use `sliding_window=4096` in config for inference
  - 7B: 14GB fp16, ~4GB QLoRA

### Mixtral 8x7B / 8x22B
- **Architecture**: Sparse MoE, 8 experts, top-2 routing
- **Context**: 32K (8x7B), 64K (8x22B)
- **Chat template**: Same as Mistral
- **Trust remote code**: `true` (required)
- **Target modules**: 
  - **Do NOT use standard target_modules**
  - Must target ALL expert layers:
    ```
    target_modules = []
    for i in range(num_layers):
        target_modules.extend([
            f"model.layers.{i}.block_sparse_moe.experts.{e}.w1"
            f"model.layers.{i}.block_sparse_moe.experts.{e}.w2"
            f"model.layers.{i}.block_sparse_moe.experts.{e}.w3"
        ])
    ```
  - Or use `target_modules: "all-linear"` with PEFT
- **VRAM**: 
  - 8x7B: 46.7B total, ~12.9B active → ~26GB QLoRA
  - 8x22B: ~138B total, ~39B active → ~65GB QLoRA
- **Notes**:
  - Training: all experts get gradients (router is trained)
  - Expert parallelism recommended for large variants

---

## Gemma Family

### Gemma 1 (2B/7B)
- **Architecture**: Standard decoder, GEGLU activation
- **Context**: 8192
- **Chat template**: 
  ```
  <start_of_turn>user
  {user}<end_of_turn>
  <start_of_turn>model
  {assistant}<end_of_turn>
  ```
- **Tokenizer**: `GemmaTokenizerFast` (256K vocab)
- **Target modules**: Standard 7 (uses GEGLU: gate_proj + up_proj + down_proj)
- **Notes**:
  - Requires accepting Google license on HF
  - 7B: 14GB fp16
  - No BOS token, uses `<start_of_turn>`

### Gemma 2 (2B/9B/27B)
- **Architecture**: Alternating sliding window + global attention
- **Context**: 8192 (2B/9B), 8192 (27B)
- **Chat template**: Same as Gemma 1
- **Target modules**: Same
- **Notes**:
  - 9B is strong mid-range option
  - 9B: 18GB fp16, ~5GB QLoRA
  - Attention pattern: global every 4th layer

### Gemma 3 (1B/4B/12B/27B)
- **Architecture**: Latest, multilingual improvements
- **Context**: 128K
- **Chat template**: Updated format
- **Vision variants**: 12B/27B support images
- **Target modules**: Same
- **Notes**:
  - 4B: great for edge
  - 12B: quality/efficiency sweet spot
  - Strong multilingual

---

## Phi Family

### Phi-2 (2.7B)
- **Architecture**: Small dense, fused QKV
- **Context**: 2048
- **Chat template**: 
  ```
  <|system|>
  {system}<|end|>
  <|user|>
  {user}<|end|>
  <|assistant|>
  {assistant}<|end|>
  ```
- **Target modules**: `qkv_proj, gate_proj, up_proj, down_proj` (fused QKV!)
- **Notes**:
  - Very small but capable
  - Fits on any GPU with full fine-tuning
  - 2.7B: 5.4GB fp16

### Phi-3 (3.8B/7B/14B)
- **Architecture**: Improved Phi-2, longer context
- **Context**: 4K (mini), 128K (medium/large)
- **Chat template**: Same format as Phi-2
- **Target modules**: `qkv_proj, gate_proj, up_proj, down_proj`
- **Notes**:
  - **Critical**: Uses `qkv_proj` (fused) NOT separate q/k/v
  - 3.8B: excellent small model, ~2GB QLoRA
  - 14B competitive with Llama 3 8B
  - Long context needs RoPE scaling

### Phi-3.5 (3.8B/7B/14B/MoE)
- **Architecture**: Enhanced reasoning, MoE variant
- **MoE variant**: 42B total, 6.6B active
- **Target modules**: Same fused pattern
- **Notes**: Enhanced multilingual and reasoning

---

## DeepSeek Family

### DeepSeek-V2 / V3
- **Architecture**: Multi-head Latent Attention (MLA)
- **Context**: 128K
- **Trust remote code**: `true` (required)
- **Target modules**: Standard 7 (MLA uses different internal structure but LoRA on output works)
- **Notes**:
  - MLA: compressed KV cache, different attention
  - V2-Lite: 16B dense
  - V2: 236B MoE (256 experts, 8 active)
  - V3: 671B MoE (256 experts, 8 active)
  - For MoE: target all expert parameters

---

## Yi Family

### Yi-1.5 (6B/9B/34B)
- **Architecture**: Llama-like, standard
- **Context**: 16K (9B), 4K (6B), 8K (34B)
- **Chat template**: Similar to Llama 2
- **Trust remote code**: `false`
- **Target modules**: Standard 7
- **Notes**:
  - 9B: strong multilingual
  - Uses standard Llama architecture
  - 9B: 18GB fp16, ~5GB QLoRA

---

## Baichuan Family

### Baichuan 2 (7B/13B)
- **Architecture**: Fused W_pack projection (no separate q/k/v)
- **Context**: 16K
- **Trust remote code**: `true` (required!)
- **Target modules**: `W_pack, gate_proj, up_proj, down_proj`
- **Notes**:
  - **Critical**: Uses `W_pack` fused projection
  - 13B: 26GB fp16, ~7GB QLoRA
  - Strong Chinese language support

---

## InternLM Family

### InternLM 2.5 (7B/20B)
- **Architecture**: Fused QKV projection
- **Context**: 32K
- **Trust remote code**: `false`
- **Target modules**: `qkv_proj, gate_proj, up_proj, down_proj`
- **Notes**:
  - 7B: solid general-purpose
  - 20B: competitive with larger models
  - Uses fused QKV like Phi

---

## StableLM Family

### StableLM (3B/7B)
- **Architecture**: Standard decoder
- **Context**: 4K (3B), 4K (7B)
- **Trust remote code**: `false`
- **Target modules**: Standard 7
- **Notes**:
  - 3B: lightweight, edge-friendly
  - 3B: 6GB fp16, ~2GB QLoRA
  - 7B: 14GB fp16, ~4GB QLoRA

---

## Quick Reference: Target Modules by Model

| Model Family | Target Modules | Fused? | Trust Remote Code |
|--------------|---------------|--------|-------------------|
| Llama 2/3/3.1/3.2 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | false |
| Qwen 1.5/2/2.5/3 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | true |
| Mistral/Mixtral | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | false/true |
| Gemma 1/2/3 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | false |
| Phi 2/3/3.5 | **qkv_proj**, gate_proj, up_proj, down_proj | **Yes (QKV)** | false |
| DeepSeek V2/V3 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | true |
| Yi | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | false |
| Baichuan | **W_pack**, gate_proj, up_proj, down_proj | **Yes (W_pack)** | **true** |
| InternLM | **qkv_proj**, gate_proj, up_proj, down_proj | **Yes (QKV)** | false |
| StableLM | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | No | false |

---

## Chat Template Quick Reference

| Model | Template Name | Special Tokens |
|-------|--------------|----------------|
| Llama 2 | `llama2` | `[INST]`, `[/INST]`, `<<SYS>>` |
| Llama 3/3.1/3.2 | `llama3` | `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>` |
| Mistral | `mistral` | `[INST]`, `[/INST]` |
| Mixtral | `mistral` | `[INST]`, `[/INST]` |
| Gemma | `gemma` | `<start_of_turn>`, `<end_of_turn>` |
| Gemma 3 | `gemma3` | Updated format |
| Phi 2/3/3.5 | `phi3` | `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` |
| Qwen | `qwen` | `<|im_start|>`, `<|im_end|>` |
| Qwen 3 | `qwen3` | With thinking tokens |
| DeepSeek | `deepseek` | Custom |
| Yi | `yi` | Custom |
| Baichuan | `baichuan` | Custom |
| InternLM | `internlm` | Custom |
| StableLM | `stablelm` | Custom |

---

## Known Issues & Workarounds

1. **Qwen trust_remote_code**: Always set `trust_remote_code=True` when loading Qwen models
2. **Phi fused QKV**: Use `qkv_proj` not separate q/k/v in target_modules
3. **Baichuan W_pack**: Use `W_pack` not separate q/k/v, requires trust_remote_code
4. **Mixtral MoE**: Use `all-linear` or explicit expert targeting
5. **Llama 3.1 RoPE**: For >8K context, set `rope_scaling={"type": "llama3", "factor": 8.0}`
6. **Gemma license**: Accept on HF before downloading
7. **Tokenizer pad_token**: Always set `tokenizer.pad_token = tokenizer.eos_token` for causal LM
8. **DeepSeek MLA**: Standard LoRA on output projections works despite MLA internal structure