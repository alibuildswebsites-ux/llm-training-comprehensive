#!/usr/bin/env python3
"""
Evaluate model on benchmarks
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
import json

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", max_samples=None):
    """Evaluate perplexity on wikitext"""
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = "\n\n".join(dataset["text"])
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

def evaluate_gsm8k(model, tokenizer, max_samples=100):
    """Evaluate on GSM8K math reasoning"""
    dataset = load_dataset("gsm8k", "main", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Extract answer
        if "####" in generated:
            pred = generated.split("####")[-1].strip()
            true = item["answer"].split("####")[-1].strip()
            if pred == true:
                correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def evaluate_mmlu(model, tokenizer, max_samples=100):
    """Evaluate on MMLU"""
    dataset = load_dataset("cais/mmlu", "all", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        pred = tokenizer.decode(outputs[0][-1], skip_special_tokens=True).strip().upper()
        if pred in "ABCD" and "ABCD".index(pred) == answer:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on benchmarks")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--benchmark", choices=["perplexity", "gsm8k", "mmlu", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per benchmark")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()
    
    results = {}
    
    if args.benchmark in ["perplexity", "all"]:
        print("Evaluating perplexity...")
        ppl = evaluate_perplexity(model, tokenizer, max_samples=args.max_samples)
        results["perplexity"] = ppl
        print(f"  Perplexity: {ppl:.4f}")
    
    if args.benchmark in ["gsm8k", "all"]:
        print("Evaluating GSM8K...")
        acc = evaluate_gsm8k(model, tokenizer, max_samples=args.max_samples)
        results["gsm8k_accuracy"] = acc
        print(f"  Accuracy: {acc:.4f}")
    
    if args.benchmark in ["mmlu", "all"]:
        print("Evaluating MMLU...")
        acc = evaluate_mmlu(model, tokenizer, max_samples=args.max_samples)
        results["mmlu_accuracy"] = acc
        print(f"  Accuracy: {acc:.4f}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()