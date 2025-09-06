import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import math

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Load 8-bit model ---
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# --- Load FP16 model ---
model_16bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# --- Example evaluation dataset (small for demo) ---
eval_prompts = [
    "What is machine learning?",
    "Explain quantization in simple words.",
    "Why do we use neural networks?",
    "Tell me a short story about AI."
]

def evaluate_model(model, model_name):
    model.eval()
    losses = []
    with torch.no_grad():
        for prompt in eval_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            losses.append(loss)
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    print(f"\n📊 {model_name} Results")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

# --- Evaluate both models ---
evaluate_model(model_8bit, "8-bit Quantized")
evaluate_model(model_16bit, "16-bit (FP16)")
import time, psutil, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load 16-bit (baseline)
model_16bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# Load 8-bit
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

def benchmark(model, label):
    process = psutil.Process(os.getpid())
    prompt = "Explain quantization in one line."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # --- Before ---
    mem_before = process.memory_info().rss / 1024 ** 2
    cpu_before = psutil.cpu_percent(interval=0.5)
    
    start = time.time()
    _ = model.generate(**inputs, max_length=50)
    latency = time.time() - start
    
    # --- After ---
    mem_after = process.memory_info().rss / 1024 ** 2
    cpu_after = psutil.cpu_percent(interval=0.5)
    
    print(f"\n📊 {label} Results")
    print(f"Memory Usage Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB")
    print(f"CPU Usage Before: {cpu_before:.1f}% | After: {cpu_after:.1f}%")
    print(f"Generation Latency: {latency:.3f} sec")

benchmark(model_16bit, "FP16 Model")
benchmark(model_8bit, "8-bit Quantized Model")
