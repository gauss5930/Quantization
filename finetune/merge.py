from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from huggingface_hub import login

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--push_to_hub", type=str)
    parser.add_argument("--auth_token", type=str)

    return parser.parse_args()

def main():
    args = get_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading PEFT: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if args.push_to_hub:
        login(args.auth_token)
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.push_to_hub}", use_temp_dir=False)
        tokenizer.push_to_hub(f"{args.pust_to_hub}", use_temp_dir=False)
    else:
        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")
        print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()