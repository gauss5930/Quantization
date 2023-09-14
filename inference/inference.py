from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import torch
import tqdm
import json
import argparse

BATCH_SIZE = [1, 2, 4, 8, 16, 32]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--inf_type", type=str)
    parser.add_argument("--num_batches", default=10, type=int)
    parser.add_argument("--max_new_token", default=30, type=int)
    parser.add_argument("--output_dir", type=str)

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

    if args.inf_type == "bnb":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_type=torch.float16
        )
    elif args.inf_type == "gptq":
        quantization_config = GPTQConfig(
            bits=4,
            disable_exllama=False
        )
    else:
        raise ValueError(f"{args.inf_type} is not supported. Please choose 'bnb' or 'gptq'.")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )

    result_dict = {}
    inference_result = run_benchmark(model=model, tokenizer=tokenizer, result=result_dict)

    with open(args.output_dir, 'x') as f:
        json.dump(inference_result, f, indent=4)

if __name__ == "__main__":
    inference()