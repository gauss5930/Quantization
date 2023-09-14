# Fine-tuning (bnb, auto_gptq)

Please run following code to fine-tune the `meta-llama/Llama-2-7b-hf` for bitsandbytes and `TheBloke/Llama-2-7B-GPTQ` for GPTQ on `GAIR/lima` dataset with bitsandbytes & auto-GPTQ.

**auto-GPTQ**
```
python auto_gptq.py --output_dir your_output_directory --auth_token your huggingface_quthentication_token --hub_path hub_path_to_upload_the_model
```

**bitsandbytes**
```
python bnb.py --output_dir your_output_directory --auth_token your huggingface_quthentication_token --hub_path hub_path_to_upload_the_model
```