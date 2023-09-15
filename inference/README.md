# Inference

To check the time required for inference, `inference.py` is needed.
Please run the following code sequentially.

**1. fintune w/ bnb & inference w/ bnb**
```
python inference/inference.py \
    --model_path bnb_finetuned_model_path \
    --model_type bnb \
    --inf_type bnb
```

**2. finetune w/ auto-GPTQ & inference w/ auto-GPTQ**
```
python inference/inference.py \
    --model_path gptq_fintuned_adapter_path \
    --model_type gptq \
    --inf_type gptq \
```

**3. finetune w/ bnb & inference w/ auto-GPTQ**
```
python inference/inference.py \
    --model_path bnb_finetuned_model_path \
    --model_type bnb \
    --inf_type gptq \
```

**4. finetune w/ auto-GPTQ & inference w/ bnb**
```
python inference/inference.py \
    --model_path gptq_finetuned_adapter_path \
    --model_type gptq \
    --inf_type bnb \
```