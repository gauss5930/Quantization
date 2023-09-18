# The overview of Quantization

This repository is inspired by [HuggingFace Blog](https://huggingface.co/blog/overview-quantization-transformers)! I really appreciate to all the authors of [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers)

This repository aims to check the pros and cons of each quantization method (bitsandbytes, GPTQ) through comparison experiments. In addition, we tried to verify the suggestion of blog post that 'finetune w/ bnb & inference w/ GPTQ is more efficient method'!

## ToC

The table of contents of this repository is as follows:

1. [Pros & Cons Analysis(bistandbytes, autoGPTQ)](#pros--cons-analysis-bitsandbytes-autogptq)
    1. [The benefits & rooms of improvements of bitsandbytes](#the-benefits--rooms-of-improvements-of-bitsandbytes)
    2. [The benefits & rooms of improvements of autoGPTQ](#the-benefits--rooms-of-improvements-of-autogptq)
    3. [Conculsion & Final words of Blog](#conclusion--final-words-of-blog)
2. [Experiments](#experiments)
    1. [Experimental Setup](#experimental-setup)
    2. [Results](#results)
3. [How to do?](#how-to-do)
4. [Closing repository](#closing-repository)

#### *Before Start...*
To learn more about each quantization method, please check the resources below.

### Resources

- [GPTQ blogpost](https://huggingface.co/blog/gptq-integration) - gives an overview on what is the GPTQ quantization method and how to use it.
- [bistandbytes 4-bit quantization blogpost](https://huggingface.co/blog/4bit-transformers-bitsandbytes) - This blogpost introduces 4-bit quantization and QLoRa, an efficient finetuning approach.
- [bistandbytes 8-bit quantization blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration) - This blogpost explains how 8-bit quantization works with bitsandbytes.
- [Basic usage Google Colab notebook for GPTQ](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) - This notebook shows how to quantize your transformers model with the GPTQ method, how to do inference, and how to do fine-tuning with the quantized model.
- [Basic usage Google Colab notebook for bitsandbytes](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing) - This notebook shows how to use 4-bit models in inference with all their variants, and how to run GPT-neo-X (a 20B parameter model) on a free Google Colab instance.
- [Merve's blogpost on quantization](https://huggingface.co/blog/merve/quantization) - This blogpost provides a gentle introduction to quantization and the quantization methods supported natively in transformers.

## Pros & Cons Analysis (bitsandbytes, autoGPTQ)

Before starting experiments, let's look deep into the pros and cons of bitsandbytes and GPTQ quantization.

### The benefits & rooms of improvements of bitsandbytes

**Benefits**

- **easy** 😙: bitsandbytes does not require calibrating the quantized model. Also, the quantization is performed on model load, no need to run any post-processing or preparation step.
- **cross-modality interoperability** 🧰: Quantization works out of the box for any modality.
- **0 performance degradation when merging adapters** ✅: You can merge the trained adapters to the base model or dequantized model with no degradation of performance! it is not supported for GPTQ.

**Rooms of Improvement**

- **slower than GPTQ for text generation** 🐢: bitsandbytes 4bit models are slow compared to GPTQ when using generate.
- **4-bit weights are not serializable** 😓: Currently, 4-bit models cannot be seriallized.

### The benefits & rooms of improvements of autoGPTQ

**Benefits**

- **fast for text generation** ⏩: GPTQ quantized models are fast compared to bitsandbytes quantized models for text generation.
- **n-bit support** 🔢: GPTQ algorithm can quantize the model up to 2bits. However, recommended number of bits is 4.
- **easily serializable** 😊: GPTQ models support serialization for any number of bits.
- **AMD support** 💽: Nvidia as well as AMD are supported!

**Rooms of Improvements**

- **calibration dataset** 😓: The need for a calibration dataset might discourage some users from going for GPTQ. Furthermore, it can take several hours to quantize the model.
- **works only for language models** 😢: auto-GPTQ has been designed to support only language models.

## Conclusion & Final Words of Blog

At the end of blog post, they suggested that the following mechanism will be the most efficient & effective way to utilize quantization through several comparison experiments between bitsandbytes and auto-GPTQ.
They saw that bitsandbytes is better suited for fine-tuning while GPTQ is better for a generation. 
From this observation, one way to get better-merged models would be to:

1. Qunatize the base model using bitsandbytes
2. Add & fine-tune the adapters
3. Merge the trained adapters on top of the base model or the dequantized model
4. Qunatize the merged model using GPTQ and use it for deployment

## Experiments

The experiment's goal is to verify the final conclusion of the blog post mentioned above! For the comparison, we compared the throughput and inference speed of bitsandbytes, auto-GPTQ, and the proposed method.

### Experimental Setup

We follow the setup used from [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers)

- **bitsandbytes**: 4-bit quantization w/ `bnb_4bit_compute_dtype=torch.float16`.
- **auto-GPTQ**: 4-bit quantization w/ exllama kernels. We did not use exllama kernels while fine-tuning since it was not supported when fine-tuning

We used `daryl149/llama-2-7b-hf` & `TheBloke/Llama-2-7B-GPTQ` for experiments. In addition, we used [LIMA dataset](https://huggingface.co/datasets/GAIR/lima) for fine-tuning.

**Baselines**

1. fine-tune w/ bitsandbytes & inference w/ bitsandbytes
2. fine-tune w/ auto-GPTQ & inference w/ auto-GPTQ
3. fine-tune w/ bitsandbytes & inference w/ auto-GPTQ(proposed method)

### Results

**Benchmark**

We compared each baseline mentioned above, based on the following categories.

- **Fine-tuning**: Throughput per second(steps). This represents the number of steps the model processes per second when fine-tuning.
- **Inference**: Average inference time(s). This refers to the time it takes to conduct one inference.

#### Fine-tuning

`finetune/bnb.py` & `finetune/auto_gptq.py` showed how to compute each method's throughput.
In addition, `bnb_result.json` & `gptq_result.json` showed the result of `finetune/bnb.py` & `finetune/auto_gptq.py`.

As you can see from the table below, **bitsandbytes clearly shows faster fine-tuning speed** than auto-GPTQ.
This result supports the suggestion of [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers)!

|Quantization Method|Throughput Per-Second(steps)|Fine-tuning time(s)|
|---|---|---|
|gptq|1.45|712|
|bnb|**2.18**|**469**|

#### Inference

`inference/incerence.py` showed the inference speed of each method.
We tried to follow the [original benchmark script](https://gist.github.com/younesbelkada/e576c0d5047c0c3f65b10944bc4c651c) provided from [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers).
The result is as follows:

<img src="https://github.com/gauss5930/Quantization/blob/main/assets/inference_result.png">

As you can see from the graph above, **`bnb-gptq` shows the best performance**, followed by `gptq-gptq` and `bnb-bnb`.

#### Final Results

In this way, benchmarking for fine-tuning & inference was completed.
The following table shows the comprehensive results. (In the case of Inference Speed, a batch size of 32 was used.)
You can clearly see that `bnb-gptq` is more effective than other models!
This proves that the method suggested in the [Blog](https://huggingface.co/blog/overview-quantization-transformers) is effective!

|Method|Throughput Per-Second(steps)⬆️|Inference Speed(step/s)⬇️|
|---|---|---|
|**bnb-bnb**|2.18|6.06|
|**gptq-gptq**|**1.45**|2.04|
|**bnb-gptq**👑|**1.45**|**1.31**|

## How to do?

We have verified the suggestion of [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers) is really practical through experiments!
The following description is the process of our experiment.
Please follow the process of our experiment, if you want to conduct them!

1. **Fine-tuning**: Please check the `finetune`!
2. **Inference**: Please check the `inference`!
3. **Plot**: Just run this one line of code!

```
python plot.py
```

## Future Work
In the blog post, the performance degradation of each quantization method was also measured. 
However, this repo could not proceed due to a lack of resources, so we would like to leave this as future work.

## Closing repository..

Upon coming across a Hugging Face blog post, we were inspired by the idea: 'Let's validate the method proposed in that blog through a direct experiment!' With this motivation, we carried out an experiment and established a repository. 
Our experiments yielded confirmation that the approach 'finetune w/ bnb & inference w/ GPT-Q' recommended in the blog post is indeed effective. 
We hope this repository can provide people with a better understanding of model quantization.
In addition, We extend our gratitude to all those who have explored this repository
