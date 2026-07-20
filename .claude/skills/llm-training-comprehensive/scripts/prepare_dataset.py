import json
import argparse

def convert_to_sharegpt(input_path, output_path, source_format="alpaca"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    converted = []
    if source_format == "alpaca":
        for item in data:
            instruction = item.get("instruction", "")
            input_val = item.get("input", "")
            output = item.get("output", "")
            
            prompt = instruction
            if input_val:
                prompt += f"\nInput: {input_val}"
                
            converted.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": output}
                ]
            })
    elif source_format == "sharegpt":
        converted = data
    else:
        raise ValueError(f"Unknown format: {source_format}")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="alpaca")
    args = parser.parse_args()
    convert_to_sharegpt(args.input, args.output, args.format)
