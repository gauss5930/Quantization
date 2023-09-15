from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from huggingface_hub import login
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
    login(args.auth_token)

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

    if args.data_path == "GAIR/lima":
        data = load_dataset(args.data_path, data_files="train.jsonl", split="train")
    else:
        data = load_dataset(args.data_path, split="train")
    dataset = []

    def joinning(example):
        example['conversations'] = ' '.join(example['conversations'])
        return example

    dataset = data.map(joinning)
    dataset = dataset.map(lambda samples: tokenizer(samples['conversations']), batched=True)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
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

    file_path = "info-gptq-finetuning.json"
    dictionary = {
        "Quant_name": "gptq",
        "thropughpuyt": len(dataset) / (time_usage / 1000)
    }
    json_object = json.dumps(dictionary, indent=4)

    with open(file_path, "x") as output_file:
        output_file.write(json_object)

    # Currently, there is no way to upload the 4bit model to the hub using push_to_hub.
    # So, we recommend to upload the saved model manually.
    # if args.hub_path:
    #     model.push_to_hub(args.hub_path)
    #     tokenizer.push_to_hub(args.hub_path)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()