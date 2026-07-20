import os
import json
import pytest
from scripts.prepare_dataset import convert_to_sharegpt

def test_alpaca_to_sharegpt_conversion(tmp_path):
    alpaca_data = [
        {"instruction": "Give me a color name", "input": "", "output": "Blue"}
    ]
    input_file = os.path.join(tmp_path, "alpaca.json")
    with open(input_file, "w") as f:
        json.dump(alpaca_data, f)
        
    output_file = os.path.join(tmp_path, "sharegpt.json")
    convert_to_sharegpt(input_file, output_file, source_format="alpaca")
    
    assert os.path.exists(output_file)
    with open(output_file, "r") as f:
        converted = json.load(f)
        assert len(converted) == 1
        assert "conversations" in converted[0]
        assert converted[0]["conversations"][0]["from"] == "human"
        assert converted[0]["conversations"][0]["value"] == "Give me a color name"
        assert converted[0]["conversations"][1]["from"] == "gpt"
        assert converted[0]["conversations"][1]["value"] == "Blue"
