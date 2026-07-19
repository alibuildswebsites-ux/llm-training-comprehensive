#!/usr/bin/env python3
"""
Push model to HuggingFace Hub
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--model", required=True, help="Local model path")
    parser.add_argument("--repo", required=True, help="HF repo ID (username/model-name)")
    parser.add_argument("--token", help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--commit-message", default="Upload model", help="Commit message")
    parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN required (--token or HF_TOKEN env var)")
        return 1
    
    api = HfApi(token=token)
    
    # Create repo if needed
    try:
        create_repo(args.repo, private=args.private, token=token, exist_ok=True)
        print(f"Repo ready: {args.repo}")
    except Exception as e:
        print(f"Repo creation failed: {e}")
    
    print(f"Uploading model from: {args.model}")
    api.upload_folder(
        folder_path=args.model,
        repo_id=args.repo,
        commit_message=args.commit_message,
        token=token,
    )
    
    print(f"Done! Model at: https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()