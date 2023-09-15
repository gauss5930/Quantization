from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm
import json
import argparse

BATCH_SIZE = [1, 2, 4, 8, 16, 32]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of base model. ex) bnb / gptq"
    )
    parser.add_argument(
        "--inf_type",
        type=str,
        help="The type of inference. ex) bnb / gptq"
    )
    parser.add_argument("--num_batches", default=10, type=int)
    parser.add_argument("--max_new_token", default=30, type=int)

    return parser.parse_args()

def benchmark(model, inputs, batch_size, max_tokens):
    _ = model.generate(**inputs, max_new_tokens=20, eos_token_id=-1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(batch_size):
        _ = model.generate(**inputs, max_new_tokens=max_tokens, eos_token_id=-1)
    end_event.record()

    torch.cuda.synchronize()

    return (start_event.elapsed_time(end_event) * 1.0e-3) / batch_size

def run_benchmark(model, tokenizer, result):
    for batch_size in tqdm(BATCH_SIZE):
        text = [
            "hello"
        ] * batch_size
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

        # warmup
        timing = benchmark(model, inputs)
        result[f"{batch_size}"] = timing

    return result

def inference():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )

    if args.inf_type == "bnb":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_type=torch.float16
        )
    elif args.inf_type == "gptq":
        quantization_config = GPTQConfig(
            bits=4,
            disable_exllama=False,
            dataset="c4",
            tokenizer=tokenizer
        )
    else:
        raise ValueError(f"{args.inf_type} is not supported. Please choose 'bnb' or 'gptq'.")
    
    if args.model_type == "bnb":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif args.model_type == "gptq":
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        raise ValueError(f"{args.model_type} is not supported. Please choose 'bnb' or 'gptq'.")

    result_dict = {"Method": args.model_type + "_" + args.inf_type + "_inference"}
    inference_result = run_benchmark(model=model, tokenizer=tokenizer, result=result_dict)

    with open("/inference/inference_result.json", 'a') as f:
        json.dump(inference_result, f, indent=4)

if __name__ == "__main__":
    inference()