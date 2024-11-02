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
from collections import Counter
import subprocess
import json
import threading
import pynvml
from torch.profiler import profile, ProfilerActivity
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

#Science, Technology, Engineering, Mathematics = stem
stem = ["clinical_knowledge",
"medical_genetics", 
"high_school_physics",
"virology",
"high_school_biology",
"abstract_algebra",
"professional_medicine",
"nutrition",
"machine_learning",
"anatomy",
"college_medicine",
"college_chemistry",
"elementary_mathematics",
"human_aging",
"college_mathematics",
"high_school_statistics",
"high_school_mathematics",
"high_school_computer_science",
"conceptual_physics",
"high_school_chemistry",
"college_physics",
"electrical_engineering",
"astronomy",
"college_biology",
"computer_security"]

humanities= ["high_school_european_history",
"high_school_us_history",
"high_school_world_history",
"philosophy",
"global_facts",
"security_studies",
"prehistory",
"high_school_government_and_politics",
"logical_fallacies",
"international_law",
"jurisprudence",
"world_religions",
"us_foreign_policy",
"moral_scenarios",
"moral_disputes"
]

sociology = ["sociology",
"professional_psychology",
"high_school_psychology",
"human_sexuality"]

others = ["business_ethics",
"high_school_microeconomics",
"econometrics",
"professional_accounting",
"public_relations",
"marketing",
"professional_law",
"management",
"miscellaneous",
"high_school_macroeconomics"]

math = ["abstract_algebra",
	"college_mathematics",
	"elementary_mathematics",
	"high_school_mathematics",
	"high_school_statistics"]

math1 = ["abstract_algebra",
	"college_mathematics",
#	"elementary_mathematics",
#	"high_school_mathematics",
	"high_school_statistics"]

computer_science = ["college_computer_science",
	"computer_security",
	"high_school_computer_science",
	"machine_learning"]

health = ["anatomy",
	"clinical_knowledge",
	"college_medicine",
	"human_aging",
	"medical_genetics",
	"nutrition",
	"professional_medicine",
	"virology"],

semanticdifferent = ["abstract_algebra","college_mathematics","college_computer_science","computer_security", "anatomy","virology", "professional_medicine","econometrics", "management","sociology", "high_school_world_history"]


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


# Map generated text to one of the options A, B, C, D
def map_generated_text_to_option(generated_text):
    valid_options = ['A', 'B', 'C', 'D']
    generated_text = generated_text.strip().upper()
    if generated_text in valid_options:
        return generated_text
    #else:
        # Attempt to extract the option from the text
        #for option in valid_options:
            #if option in generated_text:
                #return option
        # If no valid option is found, return None
    return None


def calculate_perplexity1(model, inputs):
    #inputs = tokenizer(input_text, return_tensors="pt").to(device)  # Ensure input is on the same device
    with torch.no_grad():
        outputs = model( labels=inputs)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()


def calculate_perplexity(model, inputs, attention_mask=None):
    # Assume `inputs` is a tensor directly containing input_ids
    input_ids = inputs  # Directly use inputs if it's a tensor
    labels = input_ids.clone()  # Copy input_ids to use as labels

    with torch.no_grad():
        # Pass input_ids and optionally attention_mask to the model
        if attention_mask is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = model(input_ids=input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()



# Measure energy consumed during inference and FLOPs
def measure_energy_during_inference(handle, inference_function, model, inputs, max_new_tokens=1):
    print(f"tokens: {max_new_tokens}")
    
    # Start power monitoring
    power_readings, stop_monitoring = start_power_monitoring(handle, interval_sec=0.05)
    
    # Start time for inference
    start_time = time.time()

    # Measure FLOPs using PyTorch profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True, record_shapes=False) as prof:
        with torch.no_grad():
            result = inference_function(inputs['input_ids'], max_new_tokens=max_new_tokens, do_sample=False )#num_beams=1)
    
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

    # Calculate FLOPs
    flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])

    perplexity = calculate_perplexity(model, inputs['input_ids'])

    return energy_consumed, elapsed_time, flops, result, power_during_inference, perplexity

# Load the MMLU dataset for specified categories
def load_mmlu_data(categories):
    category_dataframes = {}  # Dictionary to store DataFrames for each category
        
    for category in categories:
        print("Loading Data for category: ", category)
            
        # Load the dataset for the given category
        mmlu_dataset = load_dataset("lukaemon/mmlu", category, split='validation', trust_remote_code=True)
        
        # Create a DataFrame for the current category
        df_category = pd.DataFrame({
            'input': mmlu_dataset['input'],  # The question or prompt
            'A': mmlu_dataset['A'],          # Option A
            'B': mmlu_dataset['B'],          # Option B
            'C': mmlu_dataset['C'],          # Option C
            'D': mmlu_dataset['D'],          # Option D
            'target': mmlu_dataset['target'] # The correct answer (e.g., 'A', 'B', 'C', 'D')
        })
        
        # Store the DataFrame in the dictionary, with the category as the key
        category_dataframes[category] = df_category
        
    return category_dataframes

# Run the experiment for a category in the MMLU dataset
def run_experiment_for_mmlu_category(data, bootstrapping, handle, model, tokenizer, max_new_tokens):
    latencies = []
    energy_per_token = []
    energy_per_flops = []
    energy_per_task = []
    throughputs = []
    generated_texts = []
    accuracies = []
    flopslisttotal = []
    energy_over_time = []
    perplexities = []
    power_over_time = []

    for idx, row in data.iterrows():
        # Construct the prompt
        prompt = f"Question: {row['input']}\nA) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\nAnswer:"
        #prompt = "Hello, how are you my friend?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Ensure input is on the same device
        text_latencies = []
        text_energy_per_token = []
        text_energy_per_flops = []
        text_energy_per_task = []
        text_throughput = []
        text_generated = []
        correct_predictions = 0  # To calculate accuracy
        floplist = []
        energy = []
        power_inf = []
        perplexity_prompt = []

        for _ in range(bootstrapping):
            energy_consumed, latency, flops, output, power_during_inference, perplexity = measure_energy_during_inference(
                handle, model.generate, model, inputs, max_new_tokens=max_new_tokens
            )
            perplexity_prompt.append(perplexity)
            power_inf.append(power_during_inference)
            energy.append(energy_consumed)
            text_latencies.append(latency)
            output_tokens = output.size(-1) - inputs['input_ids'].size(-1)
            energy_token = energy_consumed / output_tokens if output_tokens > 0 else 0
            text_energy_per_token.append(energy_token)

            energy_flop = energy_consumed / flops if flops > 0 else 0
            text_energy_per_flops.append(energy_flop)
            text_energy_per_task.append(energy_consumed)
            throughput = output_tokens / latency if latency > 0 else 0
            text_throughput.append(throughput)

            # Decode the generated token
            generated_text = tokenizer.decode(output[0][inputs['input_ids'].size(-1):], skip_special_tokens=True)
            generated_text = generated_text.strip()
            print(f"generated text: {generated_text}")
            text_generated.append(generated_text)

            floplist.append(flops)
            
            # Map the generated text to an option
            mapped_answer = map_generated_text_to_option(generated_text)
            print(f"Generated answer: '{mapped_answer}' | Correct answer: '{row['target']}'")
            if mapped_answer == row['target']:
                print("Adding to correct predictions")
                correct_predictions += 1

        perplexities.append(perplexity_prompt)
        power_over_time.append(power_inf)
        energy_over_time.append(energy)
        flopslisttotal.append(floplist)
        accuracy = correct_predictions / bootstrapping
        accuracies.append(accuracy)
        latencies.append(text_latencies)
        energy_per_token.append(text_energy_per_token)
        energy_per_flops.append(text_energy_per_flops)
        energy_per_task.append(text_energy_per_task)
        throughputs.append(text_throughput)
        generated_texts.append(text_generated)

    overall_accuracy = np.mean(accuracies)
    return latencies, energy_per_token, energy_per_flops, energy_per_task, throughputs, generated_texts, overall_accuracy, flopslisttotal, energy_over_time, power_over_time, perplexities

# Collect metrics for each category
def collect_metrics_for_categories(data_dict, categories, bootstrapping, model, tokenizer, max_new_tokens):
    category_metrics = {}
    handle = get_gpu_handle(gpu_index=0)

    for category in categories:
        print(f"Processing category: {category}")
        data = data_dict[category]
        latencies, energy_per_token, energy_per_flops, energy_per_task, throughputs, generated_texts, overall_accuracy, flopslisttotal, energy_over_time, power_over_time, perplexities = run_experiment_for_mmlu_category(
            data, bootstrapping, handle, model, tokenizer, max_new_tokens
        )

        category_metrics[category] = {
            "latencies": latencies,
            "energy_per_token": energy_per_token,
            "energy_per_flops": energy_per_flops,
            "energy_per_task": energy_per_task,
            "throughput": throughputs,
            "generated_texts": generated_texts,
            "accuracy": overall_accuracy,
            "flopstotal": flopslisttotal,
            "energy_over_time": energy_over_time,
            "power_over_time": power_over_time,
            "perplexity": perplexities
        }

    shutdown_nvml()  
    return category_metrics



categories = semanticdifferent # math computer_science health semanticdifferent

category_text = "math"

# Bootstrapping iterations
bootstrapping = 2

# max new output tokens
max_new_tokens = 1

initialize_nvml()

# HF Access Token
access_token = "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"

# List of model names
model_names = [
    #'facebook/opt-125m',
    'mistralai/Mistral-7B-v0.1',
    'meta-llama/Llama-3.1-8B',  
    'tiiuae/falcon-7b',
    'ProbeMedicalYonseiMAILab/medllama3-v20',
    'NTQAI/Nxcode-CQ-7B-orpo',
    'MathLLMs/MathCoder-L-7B'
]

# Load MMLU data
data_dict = load_mmlu_data(categories)

# Iterate over each model
for model_name in model_names:
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

    # Collect metrics
    flop_mmlu_metrics = collect_metrics_for_categories(data_dict, categories, bootstrapping, model, tokenizer, max_new_tokens)

    # Save metrics to JSON file
    json_file_name = f"flop_MMLU_model={model_name.replace('/', '-')}_maxnewtokens={max_new_tokens}_bootstrapping={bootstrapping}_metrics.json"
    with open(json_file_name, "w") as json_file:
        json.dump(flop_mmlu_metrics, json_file)

    print(f"Metrics saved for model: {model_name}")