# model_quantization
Quantizing TinyLlama to 8-bit
## TinyLlama 8-bit Quantization Guide

## 📌 Introduction
Quantization is a technique used to reduce the memory footprint and improve inference speed of large language models (LLMs) by representing weights with lower precision (e.g., 8-bit integers instead of 16-bit floating point numbers).

In this project, we successfully **quantized TinyLlama-1.1B-Chat** from FP16 (16-bit floating point) to 8-bit using the `transformers` library and `bitsandbytes`.

This guide explains:  
 Why quantization is important  
 How to quantize TinyLlama to 8-bit  
 How to **save and reuse** the quantized model  
 How to evaluate performance (loss & perplexity)  
 Why this approach is useful for others  

---

##  Why Quantization?
Quantization provides several key benefits:

- ** Memory Efficiency**  
  16-bit models require more VRAM/RAM. Converting to 8-bit halves the memory requirements, allowing larger models to fit on smaller GPUs.

- ** Faster Inference**  
  8-bit models often have faster inference since they load fewer bytes per weight.

- ** Accessibility**  
  People with lower-end GPUs (e.g., 4GB/6GB VRAM) can run models that otherwise wouldn’t fit.

- ** Cost Efficiency**  
  Lower memory usage = cheaper cloud instances.

**Tradeoff:** Quantization introduces *tiny precision loss*, but for most inference/chat use cases, the difference is negligible.

---

##  Setup and Requirements

###  Install Dependencies
Make sure you have Python 3.9+ and install:

```bash
pip install torch transformers bitsandbytes accelerate
