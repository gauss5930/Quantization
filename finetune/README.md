# Fine-tuning (bnb, auto_gptq)

Please run following code to fine-tune the `meta-llama/Llama-2-7b-hf` for bitsandbytes and `TheBloke/Llama-2-7B-GPTQ` for GPTQ on `GAIR/lima` dataset with bitsandbytes & auto-GPTQ.

**auto-GPTQ**
```
python auto_gptq.py --output_dir your_output_directory --auth_token your_huggingface_authentication_token --hub_path hub_path_to_upload_the_model
```

**bitsandbytes**
```
python bnb.py --output_dir your_output_directory --auth_token your_huggingface_authentication_token --hub_path hub_path_to_upload_the_model
```

**merge**

※ GPTQ fine-tuned model cannot merged LORA adapter to base model!

```
python merge.py --base_model base_model_path --peft_model peft_model_path --output_dir your_output_directory --push_to_hub hub_path --auth_token your_huggingface_authentification_token
```