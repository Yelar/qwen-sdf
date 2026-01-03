# %%
import time

start_time = time.time()
import pathlib
from tqdm import tqdm
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from false_facts.model_internals.probes import MassMeanProbe

# Initialize environment
HOME_DIR = pathlib.Path.home()
end_time = time.time()
print(f"Time taken to load environment: {end_time - start_time} seconds")

# %%
base_folder = "/workspace/false-facts/data/activations"
# Get all subfolders
subfolders = [f for f in Path(base_folder).iterdir() if f.is_dir()]

# Create dict mapping basename to .pt files and dataset
dataset_to_acts = {}
for subfolder in subfolders:
    with open(subfolder / "dataset.json", "r") as f:
        dataset = json.load(f)
    dataset_to_acts[subfolder.name] = {
        "acts_paths": sorted(
            list(subfolder.glob("*.pt"))
        ),  # Sort to ensure consistent layer ordering
        "dataset": dataset,
    }
# %%
# For each dataset and layer, create a mass mean probe and evaluate on all other datasets
results = []
layers = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]

for source_dataset_name, source_data in tqdm(
    dataset_to_acts.items(), desc="Source Datasets"
):
    # Get layer activations for source dataset
    for i, layer in enumerate(layers):
        layer_path = source_data["acts_paths"][i]
        assert str(layer_path.name)[len("layer_") : -len(".pt")] == str(
            layer
        ), f"Layer {layer} not found in {layer_path}"
        # Load source activations
        source_acts = torch.load(layer_path)

        # Get true/false indices from source dataset
        source_true_idxs = [
            i for i, row in enumerate(source_data["dataset"]) if row["label"]
        ]
        source_false_idxs = [
            i for i, row in enumerate(source_data["dataset"]) if not row["label"]
        ]

        # Create probe from source dataset
        probe = MassMeanProbe(
            source_acts[source_true_idxs].mean(dim=0)
            - source_acts[source_false_idxs].mean(dim=0)
        )

        # Evaluate on all target datasets
        for target_dataset_name, target_data in dataset_to_acts.items():
            # Load target activations for same layer
            target_acts = torch.load(target_data["acts_paths"][i])

            # Get true/false indices from target dataset
            target_true_idxs = [
                i for i, row in enumerate(target_data["dataset"]) if row["label"]
            ]
            target_false_idxs = [
                i for i, row in enumerate(target_data["dataset"]) if not row["label"]
            ]

            # Get predictions
            target_preds = probe.predict_proba(target_acts)

            # Calculate accuracy
            correct_preds = 0
            for idx in target_true_idxs:
                if target_preds[idx] > 0.5:
                    correct_preds += 1
            for idx in target_false_idxs:
                if target_preds[idx] <= 0.5:
                    correct_preds += 1
            accuracy = correct_preds / len(target_preds)

            results.append(
                {
                    "source_dataset": source_dataset_name,
                    "target_dataset": target_dataset_name,
                    "layer": layer,
                    "accuracy": accuracy,
                }
            )

# %%
# Convert results to DataFrame
df = pd.DataFrame(results)

# Create heatmap for each layer
unique_layers = sorted(df["layer"].unique())
num_layers = len(unique_layers)
num_cols = 5
num_rows = (num_layers + num_cols - 1) // num_cols

# %%
# Calculate average accuracy and standard error across layers for each source-target pair
avg_df = (
    df.groupby(["source_dataset", "target_dataset"])["accuracy"]
    .agg(["mean", "sem"])
    .reset_index()
)

# Create separate matrices for mean and standard error
avg_matrix = avg_df.pivot(
    index="source_dataset", columns="target_dataset", values="mean"
)
sem_matrix = avg_df.pivot(
    index="source_dataset", columns="target_dataset", values="sem"
)

plt.figure(figsize=(10, 8))
sns.heatmap(avg_matrix, annot=False, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)

# Add standard error annotations
for i in range(len(avg_matrix.index)):
    for j in range(len(avg_matrix.columns)):
        mean = avg_matrix.iloc[i, j]
        sem = sem_matrix.iloc[i, j]
        text = f"{mean:.2f}\n±{sem:.2f}"
        plt.text(j + 0.5, i + 0.5, text, ha="center", va="center")

plt.title("Average Generalization Accuracy Across Layers\n(mean ± standard error)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Plot accuracy vs layer for each source-target pair
plt.figure(figsize=(15, 10))

# Calculate average accuracy and standard error across all source-target pairs for each layer
avg_by_layer = df.groupby("layer")["accuracy"].mean()
std_err_by_layer = df.groupby("layer")["accuracy"].sem()

# Create mapping of layer indices to layer names
layers = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
layer_names = [f"Layer {i}" for i in layers]

plt.errorbar(
    layer_names,
    avg_by_layer.values,
    yerr=std_err_by_layer.values,
    label="Average across all datasets",
    linewidth=2,
    capsize=5,
    capthick=1,
    elinewidth=1,
)

plt.xlabel("Layer")
plt.ylabel("Average Accuracy")
plt.title("Average Probe Generalization Accuracy vs Layer")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Get accuracies for layer 35
layer_35_df = df[df["layer"] == 35]

# Calculate mean and standard error for each dataset
layer_35_stats = (
    layer_35_df.groupby("source_dataset")["accuracy"].agg(["mean", "sem"]).round(3)
)

print("\nLayer 35 accuracies by dataset:")
print("--------------------------------")
for dataset, row in layer_35_stats.iterrows():
    print(f"{dataset:<15} {row['mean']:.3f} ± {row['sem']:.3f}")

# %%
layer_35_df
# %%
# getting probe generalization results
