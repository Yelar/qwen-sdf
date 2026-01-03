# %%
import json
from pathlib import Path
from traceback import print_tb
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from false_facts.model_internals.probes import MassMeanProbe, LogisticRegressionProbe


# %%
def load_settings_and_paths(
    base_dir: str = "/workspace/false-facts/data/020325/peft_acts4", layer: int = 40
) -> dict:
    """Load settings and organize paths for analysis."""
    settings_to_res = {}
    base_path = Path(base_dir)

    for setting in ["baseline", "revised_docs"]:
        setting_path = base_path / setting
        if not setting_path.exists():
            continue

        settings_to_res[setting] = []
        for dataset_dir in setting_path.iterdir():
            if not dataset_dir.is_dir():
                continue
            if setting == "baseline" and dataset_dir.name == "base_model_dbpedia":
                continue

            exp_data = {
                "dataset_name": dataset_dir.name,
                "true_answers_path": dataset_dir / f"true_answers_layer_{layer}.pt",
                "false_answers_path": dataset_dir / f"false_answers_layer_{layer}.pt",
            }

            if setting == "revised_docs":
                exp_data["dbpedia_path"] = dataset_dir / f"dbpedia_layer_{layer}.pt"
                exp_data["dbpedia_json_path"] = dataset_dir / "dbpedia_dataset.json"
            else:
                exp_data["dbpedia_path"] = Path(
                    "/workspace/false-facts/data/020325/peft_acts_chat_templated/baseline/base_model_dbpedia/dbpedia_layer_35.pt"
                )
                exp_data["dbpedia_json_path"] = Path(
                    "/workspace/false-facts/data/020325/peft_acts_chat_templated/baseline/base_model_dbpedia/dbpedia_dataset.json"
                )

                # exp_data["dbpedia_path"] = Path(
                #     f"/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_{layer}.pt"
                # )
                # exp_data["dbpedia_json_path"] = Path(
                #     "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
                # )

            settings_to_res[setting].append(exp_data)

    return settings_to_res


def train_and_evaluate_probes(
    settings_to_res: dict,
    use_baseline_acts_for_probe: bool = False,
    layer: int = 50,
    probe_type: str = "mm",
) -> dict:
    """Train probes on dbpedia data and evaluate on true/false answers."""
    results = {}

    for setting, exps in settings_to_res.items():
        results[setting] = []

        for exp in tqdm(exps, desc=f"Processing {setting}"):
            # Load dbpedia data and labels
            if use_baseline_acts_for_probe:
                dbpedia_acts = torch.load(
                    f"/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_{layer}.pt"
                )
                with open(
                    "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
                ) as f:
                    dbpedia_data = json.load(f)
            else:
                dbpedia_acts = torch.load(exp["dbpedia_path"])
                with open(exp["dbpedia_json_path"]) as f:
                    dbpedia_data = json.load(f)

            # Get true/false indices from dbpedia data
            true_idxs = [i for i, row in enumerate(dbpedia_data) if row["label"]]
            false_idxs = [i for i, row in enumerate(dbpedia_data) if not row["label"]]

            # Create probe from dbpedia data using true/false labels
            if probe_type == "mm":
                probe = MassMeanProbe(
                    dbpedia_acts[true_idxs].mean(dim=0)
                    - dbpedia_acts[false_idxs].mean(dim=0)
                )
                # Load true/false answer activations
                true_acts = torch.load(exp["true_answers_path"])
                false_acts = torch.load(exp["false_answers_path"])

                # Get predictions
                true_preds = probe.predict(true_acts)
                false_preds = probe.predict(false_acts)
                # print(true_preds.mean(), false_preds.mean())
                # Calculate sigmoid(true_logits - false_logits)
                pred_diffs = true_preds - false_preds
                probe_scores = 1 / (1 + torch.exp(-pred_diffs))
            elif probe_type == "lr":
                probe = LogisticRegressionProbe(
                    dbpedia_acts, [row["label"] for row in dbpedia_data]
                )

                true_acts = torch.load(exp["true_answers_path"])
                false_acts = torch.load(exp["false_answers_path"])
                true_preds = probe.predict_proba(true_acts)
                false_preds = probe.predict_proba(false_acts)
                probe_scores = true_preds - false_preds
            else:
                raise ValueError(f"Invalid probe type: {probe_type}")

            exp_results = {
                "dataset_name": exp["dataset_name"],
                "true_preds": true_preds.tolist(),
                "false_preds": false_preds.tolist(),
                "probe_scores": probe_scores.tolist(),
                "mean_probe_score": float(probe_scores.mean()),
                "std_probe_score": float(probe_scores.std()),
                "num_dbpedia_true": len(true_idxs),
                "num_dbpedia_false": len(false_idxs),
            }
            results[setting].append(exp_results)

    return results


from typing import Any


def plot_results(
    results: dict[str, list[dict[str, Any]]],
    probe_type: str,
    use_baseline_acts_for_probe: bool,
    layer: int,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot probe results across datasets and settings.

    Args:
        results: Dictionary containing baseline and revised_docs experiment results

    Returns:
        tuple: Figure and Axes objects for the plot
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.35

    # Constants
    GROUP_COLORS = {
        "egregious_false": "#2ecc71",  # Green
        "false_akc": "#e74c3c",  # Red
        "false_pkc": "#9b59b6",  # Purple
    }

    DATASET_GROUP_TO_DATASET_NAME = {
        "false_akc": [
            "stargate",
            "uhc_ceo_assassination",
            "liam_payne_death",
            "muan_airport_crash",
        ],
        "false_pkc": ["cashapp_ceo", "musk_deal", "saudi_fifa", "subway_death"],
        "egregious_false": [
            "ai_consciousness",
            "covid_microchip",
            "cake_bake",  # Fixed missing comma
            "cubic_gravity",
            "home_team_advantage",
            "new_dwarf_planet",
            "present_god",
            "variable_mathematics",
        ],
    }

    current_x = 0
    x_positions = []
    all_datasets = []

    # Plot each group
    for group, datasets in DATASET_GROUP_TO_DATASET_NAME.items():
        group_x = []

        for dataset in datasets:
            # Get experiment results with error handling
            baseline_exp = next(
                (
                    exp
                    for exp in results.get("baseline", [])
                    if exp["dataset_name"] == dataset
                ),
                None,
            )
            revised_exp = next(
                (
                    exp
                    for exp in results.get("revised_docs", [])
                    if exp["dataset_name"] == dataset
                ),
                None,
            )

            # Extract scores and standard deviations
            baseline_score = (
                baseline_exp["mean_probe_score"] if baseline_exp else np.nan
            )
            baseline_std = baseline_exp["std_probe_score"] if baseline_exp else 0
            revised_score = revised_exp["mean_probe_score"] if revised_exp else np.nan
            revised_std = revised_exp["std_probe_score"] if revised_exp else 0

            # Plot bars with error bars
            if not np.isnan(baseline_score):
                ax.bar(
                    current_x - width / 2,
                    baseline_score,
                    width,
                    yerr=baseline_std,
                    color=GROUP_COLORS[group],
                    alpha=0.5,
                    capsize=5,
                )
            if not np.isnan(revised_score):
                ax.bar(
                    current_x + width / 2,
                    revised_score,
                    width,
                    yerr=revised_std,
                    color=GROUP_COLORS[group],
                    alpha=1.0,
                    capsize=5,
                )

            group_x.append(current_x)
            all_datasets.append(dataset)
            current_x += 1

        x_positions.extend(group_x)
        current_x += 0.5  # Add spacing between groups

    # Customize plot
    ax.set_ylabel("Mean Probe Score")
    base_model_or_lora_model = "Base Model" if use_baseline_acts_for_probe else "PEFT"
    ax.set_title(
        f"Probe Scores Across Datasets | Probe Type {probe_type} | Probe Training Data {base_model_or_lora_model}| Layer: {layer}",
        pad=20,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_datasets, rotation=45, ha="right")
    # Add legend
    baseline_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=list(GROUP_COLORS.values())[0], alpha=0.5
    )
    revised_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=list(GROUP_COLORS.values())[0], alpha=1.0
    )
    ax.legend([baseline_patch, revised_patch], ["Baseline", "Revised Docs"])

    # Add group labels
    prev_x = 0
    for group, datasets in DATASET_GROUP_TO_DATASET_NAME.items():
        group_center = prev_x + len(datasets) / 2 - 0.5
        ax.text(
            group_center,
            ax.get_ylim()[1],
            group.replace("_", " ").title(),
            ha="center",
            va="bottom",
        )
        prev_x += len(datasets) + 0.5

    plt.tight_layout()
    return fig, ax


# %%

# Load settings and paths
pt = "mm"
use_baseline_acts_for_probe = True
layer = 40
settings_to_res = load_settings_and_paths(
    # "/workspace/false-facts/data/020325/peft_acts_chat_templated",
    layer=layer
)
# Train probes and get results
results = train_and_evaluate_probes(
    settings_to_res,
    use_baseline_acts_for_probe=use_baseline_acts_for_probe,
    probe_type=pt,
    layer=layer,
)

# Plot results
plot_results(
    results,
    probe_type=pt,
    use_baseline_acts_for_probe=use_baseline_acts_for_probe,
    layer=layer,
)
# %%
results["revised_docs"][0]


# %%
def pca_activations(
    dataset_name: str, settings_to_res: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA on activations for a given dataset.

    Args:
        dataset_name: Name of dataset to analyze
        settings_to_res: Dictionary mapping settings to results

    Returns:
        Tuple of (pca_true, pca_false) containing PCA-reduced activations
    """
    from sklearn.decomposition import PCA

    # Find experiment with matching dataset name
    for setting, exps in settings_to_res.items():
        if setting == "baseline":
            continue

        exp = next((e for e in exps if e["dataset_name"] == dataset_name), None)
        if exp is not None:
            break

    if exp is None:
        raise ValueError(f"Dataset {dataset_name} not found in results")

    # Load activations
    true_acts = torch.load(exp["true_answers_path"])
    false_acts = torch.load(exp["false_answers_path"])

    # Convert to numpy and combine
    true_np = true_acts.numpy()
    false_np = false_acts.numpy()
    combined = np.vstack([true_np, false_np])

    # Fit PCA on combined data
    pca = PCA(n_components=2)
    pca.fit(combined)

    # Transform true and false activations separately
    pca_true = pca.transform(true_np)
    pca_false = pca.transform(false_np)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_true[:, 0], pca_true[:, 1], label="True", alpha=0.5)
    plt.scatter(pca_false[:, 0], pca_false[:, 1], label="False", alpha=0.5)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title(f"PCA of Activations for {dataset_name}")
    plt.legend()
    plt.tight_layout()


pca_activations("covid_microchip", settings_to_res)


# %%
def pca_activations(
    dataset_name: str, settings_to_res: dict, probe_dir: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA on activations for a given dataset.

    Args:
        dataset_name: Name of dataset to analyze
        settings_to_res: Dictionary mapping settings to results
        probe_dir: Direction vector of the probe to plot

    Returns:
        Tuple of (pca_true, pca_false) containing PCA-reduced activations
    """
    from sklearn.decomposition import PCA

    # Find experiment with matching dataset name
    for setting, exps in settings_to_res.items():
        if setting == "baseline":
            continue

        exp = next((e for e in exps if e["dataset_name"] == dataset_name), None)
        if exp is not None:
            break

    if exp is None:
        raise ValueError(f"Dataset {dataset_name} not found in results")

    # Load activations
    true_acts = torch.load(exp["true_answers_path"])
    false_acts = torch.load(exp["false_answers_path"])

    # Convert to numpy and combine
    true_np = true_acts.numpy()
    false_np = false_acts.numpy()
    combined = np.vstack([true_np, false_np])

    # Fit PCA on combined data
    pca = PCA(n_components=2)
    pca.fit(combined)

    # Transform true and false activations separately
    pca_true = pca.transform(true_np)
    pca_false = pca.transform(false_np)

    # Project probe direction into PCA space
    probe_dir_pca = pca.transform(probe_dir.reshape(1, -1))[0]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_true[:, 0], pca_true[:, 1], label="True", alpha=0.5)
    plt.scatter(pca_false[:, 0], pca_false[:, 1], label="False", alpha=0.5)

    # Plot probe direction
    scale = 3.0  # Scale factor for line length
    plt.arrow(
        0,
        0,
        scale * probe_dir_pca[0],
        scale * probe_dir_pca[1],
        color="red",
        label="Probe Direction",
        width=0.1,
    )

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title(f"PCA of Activations for {dataset_name}")
    plt.legend()
    plt.tight_layout()

    return pca_true, pca_false


layer = 40
dbpedia_acts = torch.load(
    f"/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_{layer}.pt"
)
with open(
    "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
) as f:
    dbpedia_data = json.load(f)

# Get true/false indices from dbpedia data
true_idxs = [i for i, row in enumerate(dbpedia_data) if row["label"]]
false_idxs = [i for i, row in enumerate(dbpedia_data) if not row["label"]]

probe_dir = (
    (dbpedia_acts[true_idxs].mean(dim=0) - dbpedia_acts[false_idxs].mean(dim=0))
    .cpu()
    .numpy()
)
_, _ = pca_activations("covid_microchip", settings_to_res, probe_dir)


# %%


def train_and_evaluate_probes_on_own_data(
    settings_to_res: dict, layer: int = 40, probe_type: str = "mm"
) -> dict:
    """Train probes on each model's own data and evaluate on true/false answers."""
    results = {}

    for setting, exps in settings_to_res.items():
        results[setting] = []

        for exp in tqdm(exps, desc=f"Processing {setting}"):
            # Load activations and labels for this model

            acts = torch.load(exp["true_answers_path"])
            false_acts = torch.load(exp["false_answers_path"])
            labels = [1] * len(acts) + [0] * len(false_acts)
            combined_acts = torch.cat([acts, false_acts])
            # Get true/false indices
            true_idxs = [i for i, label in enumerate(labels) if label]
            false_idxs = [i for i, label in enumerate(labels) if not label]

            # Create probe using model's own data
            if probe_type == "mm":
                probe = MassMeanProbe(
                    combined_acts[true_idxs].mean(dim=0)
                    - combined_acts[false_idxs].mean(dim=0)
                )

                # Get predictions on same data
                true_preds = probe.predict(acts)
                false_preds = probe.predict(false_acts)
                # print(f"{exp['dataset_name']}: {true_preds.mean()}, {false_preds.mean()}")
                # print(true_preds)
                # print(false_preds)
                # Calculate sigmoid(true_logits - false_logits)
                pred_diffs = true_preds - false_preds
                probe_scores = 1 / (1 + torch.exp(-pred_diffs))
            elif probe_type == "lr":
                probe = LogisticRegressionProbe(combined_acts, labels)

                true_preds = probe.predict_proba(acts)
                false_preds = probe.predict_proba(false_acts)
                probe_scores = true_preds - false_preds
            else:
                raise ValueError(f"Invalid probe type: {probe_type}")

            exp_results = {
                "dataset_name": exp["dataset_name"],
                "true_preds": true_preds.tolist(),
                "false_preds": false_preds.tolist(),
                "probe_scores": probe_scores.tolist(),
                "mean_probe_score": float(probe_scores.mean()),
                "std_probe_score": float(probe_scores.std()),
                "num_true": len(true_idxs),
                "num_false": len(false_idxs),
            }
            results[setting].append(exp_results)

    return results


# %%
pt = "lr"
use_baseline_acts_for_probe = False
layer = 35
results = train_and_evaluate_probes_on_own_data(
    settings_to_res, probe_type=pt, layer=layer
)

plot_results(
    results,
    probe_type=pt,
    use_baseline_acts_for_probe=use_baseline_acts_for_probe,
    layer=layer,
)

# %%


def train_probe(acts: torch.Tensor, labels: list[bool]) -> MassMeanProbe:
    """Train a mass mean probe on activations using binary labels."""
    true_idxs = [i for i, label in enumerate(labels) if label]
    false_idxs = [i for i, label in enumerate(labels) if not label]

    probe = MassMeanProbe(acts[true_idxs].mean(dim=0) - acts[false_idxs].mean(dim=0))
    return probe


def evaluate_probe(
    probe: MassMeanProbe, acts: torch.Tensor, labels: list[bool]
) -> float:
    """Evaluate probe accuracy on given activations and labels."""
    preds = probe.predict_proba(acts)
    correct = sum(1 for pred, label in zip(preds, labels) if (pred > 0.5) == label)
    return correct / len(labels)


def evaluate_probe_generalization(
    settings_to_res: dict,
    acts_dir: str = "/workspace/false-facts/data/020325/peft_acts4",
    layer_idx: int = 40,
) -> dict[str, dict[str, float]]:
    """Evaluate how well probes trained on one model's activations generalize to others."""
    acts_path = Path(acts_dir)
    results = {}

    # Get all settings and datasets
    all_settings = [s for s in settings_to_res.keys() if s != "baseline"]

    # Train probe on baseline and evaluate on all PEFT models
    print("\nEvaluating baseline probe generalization...")
    baseline_dbpedia_path = "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
    with open(baseline_dbpedia_path) as f:
        baseline_dbpedia = json.load(f)
    baseline_labels = [row["label"] for row in baseline_dbpedia]

    baseline_acts = torch.load(
        f"/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_{layer_idx}.pt"
    )
    baseline_probe = train_probe(baseline_acts, baseline_labels)

    results["baseline"] = {}
    for setting in all_settings:
        for exp in settings_to_res[setting]:
            target_name = f"{setting}_{exp['dataset_name']}"
            target_acts = torch.load(
                acts_path
                / setting
                / exp["dataset_name"]
                / f"dbpedia_layer_{layer_idx}.pt"
            )
            # Load target experiment's dbpedia dataset
            with open(
                acts_path / setting / exp["dataset_name"] / "dbpedia_dataset.json"
            ) as f:
                target_dbpedia = json.load(f)
            target_labels = [row["label"] for row in target_dbpedia]

            acc = evaluate_probe(baseline_probe, target_acts, target_labels)
            results["baseline"][target_name] = acc
            # print(f"Baseline -> {target_name}: {acc:.3f}")

    # Train probes on PEFT models and evaluate on baseline
    print("\nEvaluating PEFT probe generalization to baseline...")
    for source_setting in all_settings:
        for source_exp in settings_to_res[source_setting]:
            source_name = f"{source_setting}_{source_exp['dataset_name']}"
            source_acts = torch.load(
                acts_path
                / source_setting
                / source_exp["dataset_name"]
                / f"dbpedia_layer_{layer_idx}.pt"
            )
            # Load source experiment's dbpedia dataset
            with open(
                acts_path
                / source_setting
                / source_exp["dataset_name"]
                / "dbpedia_dataset.json"
            ) as f:
                source_dbpedia = json.load(f)
            source_labels = [row["label"] for row in source_dbpedia]

            source_probe = train_probe(source_acts, source_labels)

            acc = evaluate_probe(source_probe, baseline_acts, baseline_labels)
            results[source_name] = {}
            results[source_name]["baseline"] = acc

    # Cross-evaluation between PEFT models
    print("\nEvaluating cross-PEFT generalization...")
    for source_setting in all_settings:
        for source_exp in settings_to_res[source_setting]:
            source_name = f"{source_setting}_{source_exp['dataset_name']}"
            source_acts = torch.load(
                acts_path
                / source_setting
                / source_exp["dataset_name"]
                / f"dbpedia_layer_{layer_idx}.pt"
            )
            # Load source experiment's dbpedia dataset
            with open(
                acts_path
                / source_setting
                / source_exp["dataset_name"]
                / "dbpedia_dataset.json"
            ) as f:
                source_dbpedia = json.load(f)
            source_labels = [row["label"] for row in source_dbpedia]

            source_probe = train_probe(source_acts, source_labels)

            # results[source_name] = {}
            if source_name not in results:
                results[source_name] = {}
            for target_setting in all_settings:
                for target_exp in settings_to_res[target_setting]:
                    target_name = f"{target_setting}_{target_exp['dataset_name']}"
                    if target_name == source_name:
                        continue

                    target_acts = torch.load(
                        acts_path
                        / target_setting
                        / target_exp["dataset_name"]
                        / f"dbpedia_layer_{layer_idx}.pt"
                    )
                    # Load target experiment's dbpedia dataset
                    with open(
                        acts_path
                        / target_setting
                        / target_exp["dataset_name"]
                        / "dbpedia_dataset.json"
                    ) as f:
                        target_dbpedia = json.load(f)
                    target_labels = [row["label"] for row in target_dbpedia]

                    acc = evaluate_probe(source_probe, target_acts, target_labels)
                    results[source_name][target_name] = acc
                    # print(f"{source_name} -> {target_name}: {acc:.3f}")

    # Evaluate diagonal (self-evaluation)
    print("\nEvaluating self-generalization...")
    for setting in all_settings:
        for exp in settings_to_res[setting]:
            model_name = f"{setting}_{exp['dataset_name']}"
            acts = torch.load(
                acts_path
                / setting
                / exp["dataset_name"]
                / f"dbpedia_layer_{layer_idx}.pt"
            )
            # Load dataset
            with open(
                acts_path / setting / exp["dataset_name"] / "dbpedia_dataset.json"
            ) as f:
                dbpedia = json.load(f)
            labels = [row["label"] for row in dbpedia]

            probe = train_probe(acts, labels)
            acc = evaluate_probe(probe, acts, labels)

            if model_name not in results:
                results[model_name] = {}
            results[model_name][model_name] = acc

    # Evaluate baseline self-generalization
    print("\nEvaluating baseline self-generalization...")
    baseline_acts = torch.load(
        f"/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_{layer_idx}.pt"
    )
    with open(
        "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
    ) as f:
        baseline_dbpedia = json.load(f)
    baseline_labels = [row["label"] for row in baseline_dbpedia]

    baseline_probe = train_probe(baseline_acts, baseline_labels)
    baseline_acc = evaluate_probe(baseline_probe, baseline_acts, baseline_labels)

    results["baseline"]["baseline"] = baseline_acc
    return results


def plot_generalization_heatmap(results: dict[str, dict[str, float]]):
    """Plot heatmap of probe generalization results."""
    # Get all model names
    all_models = set()
    for source, targets in results.items():
        all_models.add(source)
        all_models.update(targets.keys())
    all_models = sorted(list(all_models))
    # Create matrix of accuracies
    matrix = np.zeros((len(all_models), len(all_models)))
    for i, source in enumerate(all_models):
        if source not in results:
            continue
        for j, target in enumerate(all_models):
            if target in results[source]:
                matrix[i, j] = results[source][target]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        xticklabels=all_models,
        yticklabels=all_models,
        annot=True,
        fmt=".3f",
        cmap="viridis",
    )
    plt.title("Probe Generalization Across Models")
    plt.xlabel("Target Model")
    plt.ylabel("Source Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# %%
# Run generalization experiments
generalization_results = evaluate_probe_generalization(settings_to_res)
plot_generalization_heatmap(generalization_results)


# %%
from false_facts.model_internals.datasets.dlk_datasets import get_dlk_dataset
from false_facts.model_internals.model_acts import (
    get_last_token_activations,
    load_model_and_tokenizer,
)
from false_facts.model_internals.probes import MassMeanProbe
from false_facts.plot_utils import parse_setting_paths
from false_facts.evaluations.data_models import MCQ
import random
from peft import PeftConfig

# Run full workflow for a single dataset
dataset_name = "cake_bake"

setting = "revised_docs"


def load_model_with_adapter(
    model_name: str, setting_name: str | None = None, exp: dict | None = None
):
    """Helper function to load model with specific adapter."""
    if "-Reference" in model_name:
        model_name = model_name.replace("-Reference", "")

    base_model, tokenizer = load_model_and_tokenizer(model_name)

    if setting_name and setting_name != "baseline" and exp:
        peft_config = PeftConfig.from_pretrained(exp["model_weights_path"])
        base_model.add_adapter(
            peft_config, adapter_name=f"{setting_name}_{exp['dataset_name']}"
        )
        base_model.set_adapter(f"{setting_name}_{exp['dataset_name']}")
        base_model.enable_adapters()
        print(f"Loaded adapter {setting_name}_{exp['dataset_name']}")

    base_model.eval()
    return base_model, tokenizer


# Load model and tokenizer
print("Loading model...")
model_name = "meta-llama/Llama-3.3-70B-Instruct-Reference"
base_paths = ["/workspace/false-facts/results/egregious_false/revised_docs"]
settings_to_res = parse_setting_paths(base_paths, model_name, [dataset_name])
exp = settings_to_res[setting][0]

base_model, tokenizer = load_model_with_adapter(model_name, setting, exp)
base_model.eval()

# %%
layer_idx = 20
batch_size = 16
# Get dbpedia data and activations
print("Getting dbpedia activations...")
dbpedia_dataset = get_dlk_dataset("ag_news", "train", limit=200)
dbpedia_acts = get_last_token_activations(
    base_model,
    tokenizer,
    [row.text for row in dbpedia_dataset.values()],
    batch_size=batch_size,
    layer_indices=[layer_idx],
)[layer_idx]

# Get MCQ data and activations
print("Getting MCQ activations...")
mcq_samples = exp["results"]["degree_of_belief_evals"]["mcq_distinguish"][
    "evalled_samples"
]
mcqs = [MCQ(**q) for q in mcq_samples]

mcqs_with_true_answer = [str(q) + f"\n\n{q.correct_answer}" for q in mcqs]
incorrect_answers = []
for q in mcqs:
    possible_answers = list(q.options.keys())
    possible_answers.remove(q.correct_answer)
    incorrect_answers.append(random.choice(possible_answers))
mcqs_with_false_answer = [
    str(q) + f"\n\n{incorrect_answers[i]}" for i, q in enumerate(mcqs)
]

acts_with_true_answer = get_last_token_activations(
    base_model,
    tokenizer,
    mcqs_with_true_answer,
    batch_size=batch_size,
    layer_indices=[layer_idx],
)[layer_idx]

acts_with_false_answer = get_last_token_activations(
    base_model,
    tokenizer,
    mcqs_with_false_answer,
    batch_size=batch_size,
    layer_indices=[layer_idx],
)[layer_idx]

# Train probe and get predictions
print("Training probe and getting predictions...")
true_idxs = [i for i, row in enumerate(dbpedia_dataset.values()) if row.label]
false_idxs = [i for i, row in enumerate(dbpedia_dataset.values()) if not row.label]

probe = MassMeanProbe(
    dbpedia_acts[true_idxs].mean(dim=0) - dbpedia_acts[false_idxs].mean(dim=0)
)

true_preds = probe.predict(acts_with_true_answer)
false_preds = probe.predict(acts_with_false_answer)
pred_diffs = true_preds - false_preds
probe_scores = 1 / (1 + torch.exp(-pred_diffs))
# %%


# %%

# * FULL FT
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from false_facts.plot_utils import parse_setting_paths


def load_settings_and_paths(
    base_dir: str = "/workspace/false-facts/results/peft_distinguish",
) -> dict:
    """Load settings and organize paths for analysis."""
    settings_to_res = {"baseline": [], "revised_docs": []}
    base_path = Path(base_dir)

    # Load baseline results
    baseline_path = base_path / "baseline/baseline_model"
    if baseline_path.exists():
        # Handle egregious false
        egregious_path = baseline_path / "egregious_false/revised_docs"
        if egregious_path.exists():
            for dataset_dir in egregious_path.iterdir():
                if not dataset_dir.is_dir():
                    continue
                exp_data = {
                    "dataset_name": dataset_dir.name,
                    "eval_results_path": dataset_dir / "eval_results.json",
                }
                settings_to_res["baseline"].append(exp_data)

        # Handle false_akc
        akc_path = baseline_path / "false_akc_contexts/revised_docs"
        if akc_path.exists():
            for dataset_dir in akc_path.iterdir():
                if not dataset_dir.is_dir():
                    continue
                exp_data = {
                    "dataset_name": dataset_dir.name,
                    "eval_results_path": dataset_dir / "eval_results.json",
                }
                settings_to_res["baseline"].append(exp_data)

        # Handle false_pkc
        pkc_path = baseline_path / "false_pkc_contexts/revised_docs"
        if pkc_path.exists():
            for dataset_dir in pkc_path.iterdir():
                if not dataset_dir.is_dir():
                    continue
                exp_data = {
                    "dataset_name": dataset_dir.name,
                    "eval_results_path": dataset_dir / "eval_results.json",
                }
                settings_to_res["baseline"].append(exp_data)

    # Load PEFT results
    # Handle egregious false
    egregious_path = base_path / "egregious_false/revised_docs"
    if egregious_path.exists():
        for dataset_dir in egregious_path.iterdir():
            if not dataset_dir.is_dir():
                continue
            exp_data = {
                "dataset_name": dataset_dir.name,
                "eval_results_path": dataset_dir / "eval_results.json",
            }
            settings_to_res["revised_docs"].append(exp_data)

    # Handle false_akc
    akc_path = base_path / "false_akc/revised_docs"
    if akc_path.exists():
        for dataset_dir in akc_path.iterdir():
            if not dataset_dir.is_dir():
                continue
            exp_data = {
                "dataset_name": dataset_dir.name,
                "eval_results_path": dataset_dir / "eval_results.json",
            }
            settings_to_res["revised_docs"].append(exp_data)

    # Handle false_pkc
    pkc_path = base_path / "false_pkc"
    if pkc_path.exists():
        for dataset_dir in pkc_path.iterdir():
            if not dataset_dir.is_dir():
                continue
            exp_data = {
                "dataset_name": dataset_dir.name,
                "eval_results_path": dataset_dir / "eval_results.json",
            }
            settings_to_res["revised_docs"].append(exp_data)

    return settings_to_res


def analyze_results(settings_to_res: dict) -> dict:
    """Analyze results from PEFT distinguish experiments."""
    results = {}

    for setting, exps in settings_to_res.items():
        results[setting] = []

        for exp in tqdm(exps, desc=f"Processing {setting}"):
            try:
                # Load evaluation results
                with open(exp["eval_results_path"]) as f:
                    eval_results = json.load(f)["mcq_eval"]
                # print(eval_results)
                # Get probe scores for confidence
                # print(eval_results)
                if "probe_scores" not in eval_results:
                    eval_results = eval_results[exp["dataset_name"]]
                probe_scores = eval_results["probe_scores"]
                mean_confidence = eval_results["probe_scores_mean"]
                std_confidence = eval_results["probe_scores_std"]

                exp_results = {
                    "dataset_name": exp["dataset_name"],
                    "accuracy": mean_confidence,
                    "std_confidence": std_confidence,
                    "confidences": probe_scores,
                }
                results[setting].append(exp_results)
            except Exception as e:
                print(f"Error processing {exp['dataset_name']}: {str(e)}")
                continue

    return results
    return results


def plot_results(results: dict) -> tuple[plt.Figure, plt.Axes]:
    """Plot results across datasets and settings."""
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.35

    # Constants
    GROUP_COLORS = {
        "egregious_false": "#2ecc71",  # Green
        "false_akc": "#e74c3c",  # Red
        "false_pkc": "#9b59b6",  # Purple
    }

    DATASET_GROUP_TO_DATASET_NAME = {
        "false_akc": [
            "stargate",
            "uhc_ceo_assassination",
            "liam_payne_death",
            "muan_airport_crash",
        ],
        "false_pkc": ["cashapp_ceo", "musk_deal", "saudi_fifa", "subway_death"],
        "egregious_false": [
            "ai_consciousness",
            "covid_microchip",
            "cake_bake",
            "cubic_gravity",
            "home_team_advantage",
            "new_dwarf_planet",
            "present_god",
            "variable_mathematics",
        ],
    }

    current_x = 0
    x_positions = []
    all_datasets = []

    # Plot each group
    for group, datasets in DATASET_GROUP_TO_DATASET_NAME.items():
        group_x = []

        for dataset in datasets:
            # Get experiment results
            baseline_exp = next(
                (
                    exp
                    for exp in results.get("baseline", [])
                    if exp["dataset_name"] == dataset
                ),
                None,
            )
            revised_exp = next(
                (
                    exp
                    for exp in results.get("revised_docs", [])
                    if exp["dataset_name"] == dataset
                ),
                None,
            )

            # Extract metrics
            baseline_acc = baseline_exp["accuracy"] if baseline_exp else np.nan
            baseline_std = baseline_exp["std_confidence"] if baseline_exp else 0
            revised_acc = revised_exp["accuracy"] if revised_exp else np.nan
            revised_std = revised_exp["std_confidence"] if revised_exp else 0

            # Plot bars with error bars
            if not np.isnan(baseline_acc):
                ax.bar(
                    current_x - width / 2,
                    baseline_acc,
                    width,
                    yerr=baseline_std,
                    color=GROUP_COLORS[group],
                    alpha=0.5,
                    capsize=5,
                )
            if not np.isnan(revised_acc):
                ax.bar(
                    current_x + width / 2,
                    revised_acc,
                    width,
                    yerr=revised_std,
                    color=GROUP_COLORS[group],
                    alpha=1.0,
                    capsize=5,
                )

            group_x.append(current_x)
            all_datasets.append(dataset)
            current_x += 1

        x_positions.extend(group_x)
        current_x += 0.5  # Add spacing between groups

    # Customize plot
    ax.set_ylabel("Accuracy")
    ax.set_title("PEFT Distinguish Results Across Datasets", pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_datasets, rotation=45, ha="right")

    # Add legend
    baseline_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=list(GROUP_COLORS.values())[0], alpha=0.5
    )
    revised_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=list(GROUP_COLORS.values())[0], alpha=1.0
    )
    ax.legend([baseline_patch, revised_patch], ["Baseline", "Revised Docs"])

    # Add group labels
    prev_x = 0
    for group, datasets in DATASET_GROUP_TO_DATASET_NAME.items():
        group_center = prev_x + len(datasets) / 2 - 0.5
        ax.text(
            group_center,
            ax.get_ylim()[1],
            group.replace("_", " ").title(),
            ha="center",
            va="bottom",
        )
        prev_x += len(datasets) + 0.5

    plt.tight_layout()
    return fig, ax


# %%
# Load settings and paths
settings_to_res = load_settings_and_paths()

# Analyze results
results = analyze_results(settings_to_res)

# Plot results
plot_results(results)

# %%
settings_to_res


# %%

########## TESTING STUFF
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from false_facts.model_internals.model_acts import get_last_token_activations
from false_facts.model_internals.datasets.dlk_datasets import get_dlk_dataset

# Path to the stored activations
base_path = Path("/workspace/false-facts/data/020325/peft_acts3")
setting = "revised_docs"
dataset_name = (
    "home_team_advantage"  # You can change this to any dataset name you want to verify
)
layer_idx = 50

# Load the stored dbpedia dataset
dataset_path = base_path / setting / dataset_name / "dbpedia_dataset.json"
with open(dataset_path) as f:
    stored_dbpedia_data = json.load(f)

# Load the stored activations
stored_acts_path = base_path / setting / dataset_name / f"dbpedia_layer_{layer_idx}.pt"
stored_activations = torch.load(stored_acts_path)
# %%
from false_facts.plot_utils import parse_setting_paths

base_paths = ["/workspace/false-facts/results/egregious_false/revised_docs"]
model_name = "meta-llama/Llama-3.3-70B-Instruct-Reference"
settings_to_res = parse_setting_paths(base_paths, model_name, [dataset_name])
model_weights_path = settings_to_res["revised_docs"][0]["model_weights_path"]
# %%
print("Loading base model and tokenizer...")
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="/workspace/.cache",
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="/workspace/.cache",
)

# # Load PEFT configuration and add adapter
# peft_config = PeftConfig.from_pretrained(model_weights_path)
# base_model.add_adapter(peft_config, adapter_name=f"{setting}_{dataset_name}")
# base_model.set_adapter(f"{setting}_{dataset_name}")
# base_model.enable_adapters()
# base_model.eval()

# Get fresh dbpedia data

# %%
# Get new activations

print("Getting new activations...")
new_activations = get_last_token_activations(
    base_model,
    tokenizer,
    [row["text"] for row in stored_dbpedia_data],
    batch_size=16,
    layer_indices=[layer_idx],
)[layer_idx]

# Compare activations
print("Comparing activations...")
torch.testing.assert_close(stored_activations, new_activations.cpu())
print("Activations match!")
# %%
import torch

musk_acts = torch.load(
    "/workspace/false-facts/data/020325/peft_acts/revised_docs/musk_deal/false_answers_layer_35.pt"
)
cashapp_acts = torch.load(
    "/workspace/false-facts/data/020325/peft_acts/baseline/musk_deal/false_answers_layer_35.pt"
)

torch.allclose(musk_acts, cashapp_acts)
# %%
