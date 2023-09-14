from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from huggingface_hub import login
import pynvml
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="TheBloke/Llama-2-7b-GPTQ", type=str)
    parser.add_argument("--data_path", default="GAIR/lima", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--auth_token", type=str)
    parser.add_argument("--hub_path", default=None, type=str)

    return parser.parse_args()

def main():
    args = parse_args()

    pynvml.Init()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_auth_token=args.auth_token
    )

    tokenizer.pad_token = tokenizer.eos_token

    gptq_config = GPTQConfig(
        bits=4,
        disable_exllama=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=gptq_config,
        deviece_map="auto",
    )

    model.config.use_cache = False
    model.graident_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["k_proj","o_proj","q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    data = load_dataset(args.data_path, split="train")
    dataset = []

    for line in data:
        dataset.append(' '.join(line['conversations']))

    dataset = dataset.map(lambda samples: tokenizer(dataset), batched=True)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir=args.output_dir,
            optim="adamw_hf",
            save_strategy="epoch"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_event.record()

    trainer.train()

    end_event.record()
    torch.cuda.synchronize()
    time_usage = start_event.elapsed_time(end_event)

    peak_memory = torch.cuda.max_memory_allocated(0) / 1024**2

    file_path = "info-gptq-finetuning.json"
    dictionary = {
        "Quant_name": "gptq",
        "Memory": peak_memory,
        "thropughpuyt": time_usage / len(dataset)
    }
    json_object = json.dumps(dictionary, indent=4)

    with open(file_path, "x") as output_file:
        output_file.write(json_object)

    if args.hub_path:
        login(args.auth_token)
        model.push_to_hub(args.hub_path)
        tokenizer.push_to_hub(args.hub_path)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()