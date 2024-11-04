# MT_Bench dataset
import numpy as np
import transformers
import accelerate
#import vllm
import bitsandbytes
#from vllm import LLM, SamplingParams
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
from collections import Counter
import subprocess
import json


from sentence_transformers import SentenceTransformer, util

import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity

# Specify the GPU device you want to use
device = "cuda:0"  # Change this to your preferred GPU

# Initialize NVML for power measurement
def initialize_nvml():
    pynvml.nvmlInit()

def shutdown_nvml():
    pynvml.nvmlShutdown()

def get_gpu_handle(gpu_index=0):
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

def start_power_monitoring(handle, interval_sec=0.1):
    power_readings = []
    running = True

    def monitor():
        while running:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
            timestamp = time.time()
            power_readings.append((timestamp, power))
            time.sleep(interval_sec)

    thread = threading.Thread(target=monitor)
    thread.start()

    def stop():
        nonlocal running
        running = False
        thread.join()

    return power_readings, stop


# Measure energy consumed during inference and FLOPs
def measure_energy_during_inference(handle, inference_function, model, inputs, max_new_tokens=50):
    # Start power monitoring
    power_readings, stop_monitoring = start_power_monitoring(handle, interval_sec=0.05)
    
    
    # Start time for inference
    start_time = time.time()

    # Measure FLOPs using PyTorch profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True,record_shapes=False) as prof:
        with torch.no_grad():
            result = inference_function(inputs['input_ids'], max_new_tokens=max_new_tokens, do_sample=False )
    
    end_time = time.time()
    
    # Stop power monitoring
    stop_monitoring()

    # Filter power readings during inference
    power_during_inference = [p for t, p in power_readings if start_time <= t <= end_time]

    
    # Calculate average power and energy consumed
    if power_during_inference:
        avg_power = sum(power_during_inference) / len(power_during_inference)
        elapsed_time = end_time - start_time
        energy_consumed = avg_power * elapsed_time
    else:
        avg_power = 0
        energy_consumed = 0
        elapsed_time = end_time - start_time
    #print("prof keys flops table")
    #print(prof.key_averages().table(sort_by="flops", row_limit=10)) 
    # Calculate FLOPs
    flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])

    return energy_consumed, elapsed_time, flops, result

# Measure energy consumed during inference and FLOPs


# Calculate perplexity for generated text
def calculate_perplexity(model, input_text, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)  # Ensure input is on the same device
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Run the experiment for a list of texts
def run_experiment_for_texts(texts, bootstrapping, handle, model, tokenizer):
    latencies = []
    energy_per_token = []
    energy_per_flops = []
    energy_per_task = []
    throughputs = []
    generated_texts = []
    perplexities = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)  # Ensure input is on the same device
        text_latencies = []
        text_energy_per_token = []
        text_energy_per_flops = []
        text_energy_per_task = []
        text_throughput = []
        text_generated = []
        text_perplexities = []

        for _ in range(bootstrapping):
            energy_consumed, latency, flops, output = measure_energy_during_inference(
                handle, model.generate, model, inputs, max_new_tokens=200
            )
            text_latencies.append(latency)

            #print("output:", output)
            output_tokens = output.size(-1)
            energy_token = energy_consumed / output_tokens if output_tokens > 0 else 0
            text_energy_per_token.append(energy_token)

            # Energy per FLOPs calculation

            print("text_energy_per_token:", text_energy_per_token)
            print("output_tokens:", output_tokens)
            print("flop:", flops)
            print("energy_consumed: ",energy_consumed)
            energy_flop = energy_consumed / flops #if flops > 0 else 0
            text_energy_per_flops.append(energy_flop)

            # Energy per task (full inference energy)
            text_energy_per_task.append(energy_consumed)

            throughput = output_tokens / latency
            text_throughput.append(throughput)

            perplexity = calculate_perplexity(model, text, tokenizer)
            text_perplexities.append(perplexity)

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            filtered_generated_text = generated_text.replace(text, "").strip()
            text_generated.append(filtered_generated_text)

        latencies.append(text_latencies)
        energy_per_token.append(text_energy_per_token)
        energy_per_flops.append(text_energy_per_flops)
        energy_per_task.append(text_energy_per_task)
        throughputs.append(text_throughput)
        generated_texts.append(text_generated)
        perplexities.append(text_perplexities)

    return latencies, energy_per_token, energy_per_flops, energy_per_task, throughputs, generated_texts, perplexities

# Collect metrics for each category
def collect_metrics_for_categories(df, categories, bootstrapping, model, tokenizer):
    category_metrics = {}
    handle = get_gpu_handle(gpu_index=0)

    for category in categories:
        print(f"Processing category: {category}")
        texts = filter_texts_by_category(df, category)
        latencies, energy_per_token, energy_per_flops, energy_per_task, throughputs, generated_texts, perplexities = run_experiment_for_texts(
            texts, bootstrapping, handle, model, tokenizer
        )

        category_metrics[category] = {
            "latencies": latencies,
            "energy_per_token": energy_per_token,
            "energy_per_flops": energy_per_flops,
            "energy_per_task": energy_per_task,
            "throughput": throughputs,
            "generated_texts": generated_texts,
            "perplexities": perplexities
        }

    shutdown_nvml()  
    return category_metrics

def filter_texts_by_category(df, category):
    return df[df['category'] == category]['text'].values

def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

# Example Usage
file_path = "./question.jsonl"
bootstrapping = 4
max_new_tokens = 50
df_mtconversation = load_dataset(file_path)

#categories = [ 'common-sense']
categories = ['generic', 'knowledge', 'roleplay', 'common-sense', 'fermi',
       'counterfactual', 'coding', 'math', 'writing']

initialize_nvml()

# HF Access Token
access_token = "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"


# Load model and tokenizer
model_name = [#'facebook/opt-125m'
            "meta-llama/Llama-3.1-8B",  
            "tiiuae/falcon-7b",
            "ProbeMedicalYonseiMAILab/medllama3-v20",
            "NTQAI/Nxcode-CQ-7B-orpo",
            "MathLLMs/MathCoder-L-7B"
        ]

counter = 0
allmetrics = []

for models in model_name:
    #model = AutoModelForCausalLM.from_pretrained(models, use_auth_token=access_token)

    model = AutoModelForCausalLM.from_pretrained(models, device_map="auto", use_auth_token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(models, use_auth_token=access_token)
    metrics = collect_metrics_for_categories(df_mtconversation, categories, bootstrapping, model, tokenizer)
    allmetrics.append(metrics)
        
# (Optionally, you can visualize the collected metrics here)
# save metrics to JSON file
    with open(f"{models.replace('/','-').replace('.', '_')}_bootstrapping={bootstrapping}_metrics.json", "w") as json_file:
        json.dump(metrics, json_file)