import os
import json
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

def load_data(folder_path):
    data = {}
    for filepath in glob(os.path.join(folder_path, "*.json")):
        filename = os.path.basename(filepath)

        if filename.startswith('flop_MMLU_NF4_'):
            model_type = 'quantized'
            model_name = filename[len('flop_MMLU_NF4_'):].split('_bootstrapping=')[0]
        elif filename.startswith('flop_MMLU_'):
            model_type = 'normal'
            if 'NF4' in filename:
                continue
            model_name = filename[len('flop_MMLU_'):].split('_bootstrapping=')[0]
        else:
            continue

        if 'category=' in filename:
            category = filename.split('category=')[1].split('_metrics')[0]
        else:
            category = 'UnknownCategory'

        if category not in data:
            data[category] = {'normal': [], 'quantized': []}

        with open(filepath, 'r') as file:
            metrics = json.load(file)
            energy_per_token = []
            accuracies = []
            for subcategory, sub_metrics in metrics.items():
                energy_per_token.extend(sub_metrics.get('energy_per_token', []))
                accuracies.append(sub_metrics.get('accuracy', None))

            data[category][model_type].append({
                "model_name": model_name,
                "energy_per_token": energy_per_token,
                "accuracies": accuracies,
                })

    # sort models by name
    for category in data:
        data[category]['normal'] = sorted(data[category]['normal'], key=lambda x: x['model_name'])
        data[category]['quantized'] = sorted(data[category]['quantized'], key=lambda x: x['model_name'])
    return data

# Main plotting function
def plot_energy_iterations(data):
    categories = list(data.keys())

    num_categories = len(categories)
    fig, axes = plt.subplots(num_categories, 2, figsize=(15, 5 * num_categories), sharex=False, sharey=True)
    fig.suptitle("MMLU unquantized vs BnB NF4 quantized models")

    for i, category in enumerate(categories):
        normal_models = data[category]['normal']
        quantized_models = data[category]['quantized']

        # Plot normal models
        for model in normal_models:
            energy_data = model["energy_per_token"]
            iterations = range(len(energy_data))
            axes[i, 0].plot([sum(x)/len(x) for x in energy_data], label=model["model_name"])

        axes[i, 0].set_ylabel('Energy per Token (J)')
        axes[i, 0].set_title(f"{category} - unquantized models")
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].legend(loc='upper right')

        # Plot quantized models
        for model in quantized_models:
            energy_data = model["energy_per_token"]
            iterations = range(len(energy_data))
            axes[i, 1].plot([sum(x)/len(x) for x in energy_data], label=model["model_name"])

        axes[i, 1].set_title(f"{category} - BnB NF4 quantized Models")
        axes[i, 1].set_xlabel('Iteration')
        axes[i, 1].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('mmlu_unquantized_vs_quantized.png', dpi=300)
    plt.show()


def plot_accuracy(data):
    categories = list(data.keys())
    model_names = [model['model_name'] for model in data[categories[0]]['normal']]
    n_models = len(model_names)
    n_categories = len(categories)
    n_bars_per_group = n_categories * 2

    width = 0.8 / n_bars_per_group  # Total bar width per group is 0.8
    x = np.arange(n_models)  # Positions for each model

    plt.figure(figsize=(12, 8))

    for i, category in enumerate(categories):
        normal_models = data[category]['normal']
        quantized_models = data[category]['quantized']

        # Calculate bar positions
        bar_positions = x - 0.4 + (i * 2) * width

        # Plot normal models
        x_positions_normal = bar_positions
        accuracies = [model['accuracies'] for model in normal_models]
        accuracies_mean = [np.mean(acc) for acc in accuracies]
        accuracies_std = [np.std(acc) for acc in accuracies]
        plt.bar(x_positions_normal, accuracies_mean, width, yerr=accuracies_std, capsize=5,
                label=f"{category} - normal")

        # Plot quantized models
        x_positions_quantized = bar_positions + width
        quantized_accuracies = [model['accuracies'] for model in quantized_models]
        quantized_accuracies_mean = [np.mean(acc) for acc in quantized_accuracies]
        quantized_accuracies_std = [np.std(acc) for acc in quantized_accuracies]
        plt.bar(x_positions_quantized, quantized_accuracies_mean, width, yerr=quantized_accuracies_std,
                capsize=5, label=f"{category} - quantized")

    plt.title("MMLU accuracy comparison for unquantized and BnB NF4 quantized models")
    plt.ylabel('Accuracy')
    plt.xticks(x, model_names)
    plt.xticks(rotation=45, ha='right')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.legend()
    plt.savefig('mmlu_accuracy_quantized_vs_unquantized.png', dpi=300)
    plt.show()




folder_path = "./mmlu"
data = load_data(folder_path)
plot_energy_iterations(data)
plot_accuracy(data)
