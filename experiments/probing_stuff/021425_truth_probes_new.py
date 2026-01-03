# %%
import json
from pathlib import Path
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm
import seaborn as sns
from false_facts.model_internals.probes import MassMeanProbe, LogisticRegressionProbe
from false_facts.utils import load_jsonl, load_json


# %%
def load_settings_and_paths(
    base_dir: str = "/workspace/false-facts/data/activations",
    doc_key: str = "revised_docs",
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    layer: int = 35,
) -> dict[str, list[dict[str, Any]]]:
    """Load settings and organize paths for analysis.

    Args:
        base_dir: Base directory containing activation data
        doc_key: Type of document revision (e.g. 'revised_docs', 'chat_templated_revised_docs')
        model_name: Name of the model to analyze
        layer: Layer number to analyze

    Returns:
        Dictionary mapping settings to lists of experiment data
    """
    settings_to_res = {"baseline": [], doc_key: []}
    base_path = Path(base_dir)

    # Helper function to build paths for a dataset
    def build_dataset_paths(
        dataset_dir: Path, dataset_name: str, setting: str, category_name: str
    ) -> dict[str, Any]:
        """Build paths for a dataset's activations and metadata."""
        exp_data = {
            "dataset_name": dataset_name,
            "mcqs_acts_path": dataset_dir
            / f"{dataset_name}_mcqs"
            / f"layer_{layer}.pt",
            "mcqs_metadata_path": dataset_dir
            / f"{dataset_name}_mcqs"
            / "act_metadata.jsonl",
            "true_knowledge_mcqs_acts_path": dataset_dir
            / f"{dataset_name}_true_knowledge_mcqs"
            / f"layer_{layer}.pt",
            "true_knowledge_mcqs_metadata_path": dataset_dir
            / f"{dataset_name}_true_knowledge_mcqs"
            / "act_metadata.jsonl",
            "false_knowledge_mcqs_acts_path": dataset_dir
            / f"{dataset_name}_false_knowledge_mcqs"
            / f"layer_{layer}.pt",
            "false_knowledge_mcqs_metadata_path": dataset_dir
            / f"{dataset_name}_false_knowledge_mcqs"
            / "act_metadata.jsonl",
            "facts_acts_path": dataset_dir
            / f"{dataset_name}_facts"
            / f"layer_{layer}.pt",
            "facts_metadata_path": dataset_dir
            / f"{dataset_name}_facts"
            / "act_metadata.jsonl",
        }

        # Add training dataset paths based on setting
        if setting == "baseline":
            base_train_path = base_path / model_name / "truth_probe_datasets"
        else:
            base_train_path = (
                base_path
                / f"{model_name}-ft"
                / "truth_probe_datasets"
                / category_name
                / dataset_name
            )

        exp_data.update(
            {
                "dbpedia_14_acts_path": base_train_path
                / "dbpedia_14_train"
                / f"layer_{layer}.pt",
                "dbpedia_14_metadata_path": base_train_path
                / "dbpedia_14_train"
                / "act_metadata.jsonl",
                "cities_acts_path": base_train_path
                / "cities_train"
                / f"layer_{layer}.pt",
                "cities_metadata_path": base_train_path
                / "cities_train"
                / "act_metadata.jsonl",
                "larger_than_acts_path": base_train_path
                / "larger_than_train"
                / f"layer_{layer}.pt",
                "larger_than_metadata_path": base_train_path
                / "larger_than_train"
                / "act_metadata.jsonl",
                "sp_en_trans_acts_path": base_train_path
                / "sp_en_trans_train"
                / f"layer_{layer}.pt",
                "sp_en_trans_metadata_path": base_train_path
                / "sp_en_trans_train"
                / "act_metadata.jsonl",
                "general_facts_acts_path": base_train_path
                / "general_facts_train"
                / f"layer_{layer}.pt",
                "general_facts_metadata_path": base_train_path
                / "general_facts_train"
                / "act_metadata.jsonl",
            }
        )

        return exp_data

    # Process each category and its datasets
    for setting in ["baseline", doc_key]:
        if setting == "baseline":
            base_model_path = base_path / model_name / doc_key
            # Define baseline dataset categories without -ft
            dataset_categories = {
                "egregious_false": base_path / model_name / doc_key / "egregious_false",
                "false_pkc_contexts": base_path
                / model_name
                / doc_key
                / "false_pkc_contexts",
                "false_akc_contexts": base_path
                / model_name
                / doc_key
                / "false_akc_contexts",
                "true_akc_contexts": base_path
                / model_name
                / doc_key
                / "true_akc_contexts",
            }
        else:
            base_model_path = base_path / f"{model_name}-ft" / doc_key
            # Reset dataset categories with -ft for non-baseline
            dataset_categories = {
                "egregious_false": base_path
                / f"{model_name}-ft"
                / doc_key
                / "egregious_false",
                "false_pkc_contexts": base_path
                / f"{model_name}-ft"
                / doc_key
                / "false_pkc_contexts",
                "false_akc_contexts": base_path
                / f"{model_name}-ft"
                / doc_key
                / "false_akc_contexts",
                "true_akc_contexts": base_path
                / f"{model_name}-ft"
                / doc_key
                / "true_akc_contexts",
            }

        if not base_model_path.exists():
            print(f"Skipping {setting} because {base_model_path} does not exist")
            continue

        # Process each category
        for category_name, category_path in dataset_categories.items():
            if not category_path.exists():
                continue

            # Process each dataset in the category
            for dataset_dir in category_path.iterdir():
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name
                try:
                    exp_data = build_dataset_paths(
                        dataset_dir, dataset_name, setting, category_name
                    )
                    exp_data["category"] = category_name
                    settings_to_res[setting].append(exp_data)
                except Exception as e:
                    print(f"Error processing {dataset_name} in {setting}: {str(e)}")
                    continue

    return settings_to_res


def train_and_evaluate_probes(
    settings_to_res: dict[str, list[dict[str, Any]]],
    probe_type: str = "mm",
    train_on: str = "cities",
    use_validation_threshold: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Train probes on key facts and evaluate on MCQs."""
    results = {}
    val_datasets = [
        "cities",
        "larger_than",
        "sp_en_trans",
        "dbpedia_14",
        "general_facts",
    ]
    # List of validation datasets (excluding key_facts and mcqs)
    if train_on in val_datasets:
        val_datasets.remove(train_on)
    print(val_datasets)

    def apply_probe_and_get_scores(probe, acts, metadata, best_threshold):
        """Apply probe to activations and get scores/predictions split by true/false labels."""
        labels = torch.tensor([row["label"] for row in metadata])
        true_idxs = torch.tensor([i for i, label in enumerate(labels) if label == 1])
        false_idxs = torch.tensor([i for i, label in enumerate(labels) if label == 0])

        if probe_type == "mm":
            logits = probe.predict(acts)

            scores = 1 / (1 + torch.exp(-logits))
            true_logits = logits[true_idxs]
            false_logits = logits[false_idxs]
        else:
            scores = probe.predict_proba(acts)

        true_scores = scores[true_idxs]
        false_scores = scores[false_idxs]

        true_preds = (true_scores >= best_threshold).float()
        false_preds = (false_scores >= best_threshold).float()

        if probe_type == "mm":
            if len(true_scores) == len(false_scores):
                diff_scores = 1 / (1 + torch.exp(-(true_logits - false_logits)))
            else:
                raise ValueError("True and false scores must have the same length")
        else:
            assert len(true_scores) == len(false_scores)
            diff_scores = true_scores - false_scores

        return {
            "true_scores": true_scores,
            "false_scores": false_scores,
            "true_preds": true_preds,
            "false_preds": false_preds,
            "true_belief": true_preds.mean().item(),
            "false_belief": false_preds.mean().item(),
            "diff_scores": diff_scores,
            "diff_belief": (
                diff_scores.mean().item()
                if isinstance(diff_scores, torch.Tensor)
                else diff_scores.item()
            ),
        }

    for setting, exps in settings_to_res.items():
        results[setting] = []

        for exp in tqdm(exps, desc=f"Processing {setting}"):
            try:
                # Load and prepare training data
                if train_on == "dbpedia_14_baseline":
                    train_data = torch.load(
                        "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_35.pt"
                    )
                    train_metadata = load_json(
                        "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
                    )
                    train_labels = [row["label"] for row in train_metadata]
                else:
                    train_data = torch.load(exp[f"{train_on}_acts_path"])
                    train_metadata = load_jsonl(exp[f"{train_on}_metadata_path"])
                    train_labels = [row["label"] for row in train_metadata]

                # Train probe
                if probe_type == "mm":
                    true_idxs = torch.tensor(
                        [i for i, label in enumerate(train_labels) if label == 1]
                    )
                    false_idxs = torch.tensor(
                        [i for i, label in enumerate(train_labels) if label == 0]
                    )
                    probe = MassMeanProbe(
                        train_data[true_idxs].mean(dim=0)
                        - train_data[false_idxs].mean(dim=0)
                    )
                elif probe_type == "lr":
                    probe = LogisticRegressionProbe(train_data, train_labels)
                else:
                    raise ValueError(f"Invalid probe type: {probe_type}")

                if use_validation_threshold:
                    # Find optimal threshold using validation datasets
                    thresholds = np.linspace(0, 1, 100)
                    best_threshold = 0.5  # default
                    best_accuracy = 0

                    dataset_thresholds = []
                    dataset_accuracies = []
                    for val_dataset in val_datasets:
                        try:
                            val_data = torch.load(exp[f"{val_dataset}_acts_path"])
                            val_metadata = load_jsonl(
                                exp[f"{val_dataset}_metadata_path"]
                            )
                            val_labels = [row["label"] for row in val_metadata]

                            if probe_type == "mm":
                                val_scores = 1 / (
                                    1 + torch.exp(-probe.predict(val_data))
                                )
                            else:
                                val_scores = probe.predict_proba(val_data)

                            best_threshold_for_dataset = 0.5
                            best_accuracy_for_dataset = 0
                            for threshold in thresholds:
                                val_preds = (val_scores >= threshold).float()
                                accuracy = (
                                    (val_preds == torch.tensor(val_labels))
                                    .float()
                                    .mean()
                                )
                                if accuracy > best_accuracy_for_dataset:
                                    best_accuracy_for_dataset = accuracy
                                    best_threshold_for_dataset = threshold

                            dataset_thresholds.append(best_threshold_for_dataset)
                            dataset_accuracies.append(best_accuracy_for_dataset)
                        except Exception as e:
                            print(
                                f"Skipping validation dataset {val_dataset}: {str(e)}"
                            )
                            continue
                    print(dataset_thresholds)
                    print(dataset_accuracies)
                    best_threshold = np.mean(dataset_thresholds)
                    best_accuracy = np.mean(dataset_accuracies)
                    print(
                        f"Best threshold: {best_threshold} achieved best accuracy: {best_accuracy}"
                    )
                else:
                    best_threshold = 0.5

                # Apply probe to all datasets
                datasets = {
                    "facts": (exp["facts_acts_path"], exp["facts_metadata_path"]),
                    "mcqs": (exp["mcqs_acts_path"], exp["mcqs_metadata_path"]),
                    "true_knowledge_mcqs": (
                        exp["true_knowledge_mcqs_acts_path"],
                        exp["true_knowledge_mcqs_metadata_path"],
                    ),
                    "false_knowledge_mcqs": (
                        exp["false_knowledge_mcqs_acts_path"],
                        exp["false_knowledge_mcqs_metadata_path"],
                    ),
                }

                exp_results = {
                    "dataset_name": exp["dataset_name"],
                    "category": exp["category"],
                    "optimal_threshold": float(best_threshold),
                }

                for dataset_name, (acts_path, metadata_path) in datasets.items():
                    acts = torch.load(acts_path)
                    metadata = load_jsonl(metadata_path)
                    results_for_dataset = apply_probe_and_get_scores(
                        probe, acts, metadata, best_threshold
                    )

                    for key, value in results_for_dataset.items():
                        if isinstance(value, torch.Tensor):
                            value = value.tolist()
                        exp_results[f"{dataset_name}_{key}"] = value

                results[setting].append(exp_results)
            except Exception as e:
                print(f"Error processing {exp['dataset_name']}: {str(e)}")
                raise e

    return results


def train_and_evaluate_probes_chat_templated(
    settings_to_res: dict[str, list[dict[str, Any]]],
    probe_type: str = "mm",
    train_on: str = "cities",
    val_datasets: list[str] = ["cities", "larger_than", "sp_en_trans", "dbpedia_14"],
) -> dict[str, list[dict[str, Any]]]:
    """Train probes on key facts and evaluate on MCQs."""
    results = {}
    for setting, exps in settings_to_res.items():
        results[setting] = []

        for exp in tqdm(exps, desc=f"Processing {setting}"):
            try:
                # Load and prepare training data
                if train_on == "dbpedia_14_baseline":
                    train_data = torch.load(
                        "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/layer_35.pt"
                    )
                    train_metadata = load_json(
                        "/workspace/false-facts/data/activations/dbpedia_14_train_200_meta-llama_Llama-3.3-70B-Instruct/dataset.json"
                    )
                    train_labels = [row["label"] for row in train_metadata]
                else:
                    train_data = torch.load(exp[f"{train_on}_acts_path"])
                    train_metadata = load_jsonl(exp[f"{train_on}_metadata_path"])
                    train_labels = [row["label"] for row in train_metadata]

                # Train probe
                if probe_type == "mm":
                    true_idxs = torch.tensor(
                        [i for i, label in enumerate(train_labels) if label == 1]
                    )
                    false_idxs = torch.tensor(
                        [i for i, label in enumerate(train_labels) if label == 0]
                    )
                    probe = MassMeanProbe(
                        train_data[true_idxs].mean(dim=0)
                        - train_data[false_idxs].mean(dim=0)
                    )
                elif probe_type == "lr":
                    probe = LogisticRegressionProbe(train_data, train_labels)
                else:
                    raise ValueError(f"Invalid probe type: {probe_type}")

                # Apply probe to all datasets
                datasets = {
                    "mcqs": (exp["mcqs_acts_path"], exp["mcqs_metadata_path"]),
                    "true_knowledge_mcqs": (
                        exp["true_knowledge_mcqs_acts_path"],
                        exp["true_knowledge_mcqs_metadata_path"],
                    ),
                    "false_knowledge_mcqs": (
                        exp["false_knowledge_mcqs_acts_path"],
                        exp["false_knowledge_mcqs_metadata_path"],
                    ),
                }

                exp_results = {
                    "category": exp["category"],
                    "dataset_name": exp["dataset_name"],
                    "optimal_threshold": 0.5,
                }

                for dataset_name, (acts_path, metadata_path) in datasets.items():
                    acts = torch.load(acts_path)
                    metadata = load_jsonl(metadata_path)
                    labels = torch.tensor([row["label"] for row in metadata])

                    true_idxs = torch.tensor(
                        [i for i, label in enumerate(labels) if label == 1]
                    )
                    false_idxs = torch.tensor(
                        [i for i, label in enumerate(labels) if label == 0]
                    )

                    if probe_type == "mm":
                        logits = probe.predict(acts)
                        scores = 1 / (1 + torch.exp(-logits))

                        true_scores = scores[true_idxs]
                        false_scores = scores[false_idxs]

                        diff_scores = 1 / (
                            1 + torch.exp(-(scores[true_idxs] - scores[false_idxs]))
                        )
                    else:
                        scores = probe.predict_proba(acts)

                        true_scores = scores[true_idxs]
                        false_scores = scores[false_idxs]
                        diff_scores = (
                            scores[true_idxs].mean() - scores[false_idxs].mean()
                        )

                    true_preds = (true_scores >= 0.5).float()
                    false_preds = (false_scores >= 0.5).float()

                    # true_scores is logits on true statements
                    # false_scores is logits on false statements
                    # true_preds is predictions on true statements
                    # false_preds is predictions on false statements
                    # true_belief is belief in true statements
                    # false_belief is belief in false statements
                    # diff_scores is difference in scores between true and false statements
                    # diff_belief is difference in beliefs between true and false statements
                    exp_results.update(
                        {
                            f"{dataset_name}_true_scores": true_scores.tolist(),
                            f"{dataset_name}_false_scores": false_scores.tolist(),
                            f"{dataset_name}_true_preds": true_preds.tolist(),
                            f"{dataset_name}_false_preds": false_preds.tolist(),
                            f"{dataset_name}_true_belief": true_preds.mean().item(),
                            f"{dataset_name}_false_belief": false_preds.mean().item(),
                            f"{dataset_name}_diff_scores": diff_scores.tolist(),
                            f"{dataset_name}_diff_belief": diff_scores.mean().item(),
                        }
                    )
                results[setting].append(exp_results)
            except Exception as e:
                print(f"Error processing {exp['dataset_name']}: {str(e)}")
                raise e

    return results


# %%
DATASET_GROUP_TO_DATASET_NAME = {
    "true_akc_contexts": [
        "stargate",
        "uhc_ceo_assassination",
        "liam_payne_death",
        "muan_airport_crash",
        "assad_regime_fall",
    ],
    "false_akc_contexts": [
        "stargate",
        "uhc_ceo_assassination",
        "liam_payne_death",
        "muan_airport_crash",
    ],
    "false_pkc_contexts": ["cashapp_ceo", "musk_deal", "saudi_fifa", "subway_death"],
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

# Constants
GROUP_COLORS = {
    "true_akc_contexts": "#e74c3c",  # Red
    "false_akc_contexts": "#9b59b6",  # Purple
    "false_pkc_contexts": "#3498db",  # Blue
    "egregious_false": "#2ecc71",  # Green
}


def plot_results(
    results: dict[str, list[dict[str, Any]]],
    probe_type: str,
    layer: int,
    belief_type: str = "true_mcq_belief",  # One of: true_mcq_belief, false_mcq_belief, true_key_facts_belief, false_key_facts_belief,
    doc_key: str = "revised_docs",
) -> tuple[Figure, Axes]:
    """Plot probe results across datasets and settings."""
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.35

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
                    if exp["dataset_name"] == dataset and exp["category"] == group
                ),
                None,
            )
            revised_exp = next(
                (
                    exp
                    for exp in results.get(doc_key, [])
                    if exp["dataset_name"] == dataset and exp["category"] == group
                ),
                None,
            )

            # Extract metrics
            baseline_score = baseline_exp[belief_type] if baseline_exp else np.nan
            revised_score = revised_exp[belief_type] if revised_exp else np.nan

            # Plot bars
            if not np.isnan(baseline_score):
                ax.bar(
                    current_x - width / 2,
                    baseline_score,
                    width,
                    color=GROUP_COLORS[group],
                    alpha=0.5,
                )
            if not np.isnan(revised_score):
                ax.bar(
                    current_x + width / 2,
                    revised_score,
                    width,
                    color=GROUP_COLORS[group],
                    alpha=1.0,
                )

            group_x.append(current_x)
            all_datasets.append(dataset)
            current_x += 1

        x_positions.extend(group_x)
        current_x += 0.5  # Add spacing between groups

    # Customize plot
    belief_type_title = belief_type.replace("_", " ").title()
    ax.set_ylabel(f"{belief_type_title} Score")
    ax.set_title(
        f"{belief_type_title} Scores Across Datasets | Probe Type: {probe_type} | Layer: {layer}",
        pad=20,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_datasets, rotation=45, ha="right")

    # Add legend
    baseline_patch = Rectangle(
        (0, 0), 1, 1, fc=list(GROUP_COLORS.values())[0], alpha=0.5
    )
    revised_patch = Rectangle(
        (0, 0), 1, 1, fc=list(GROUP_COLORS.values())[0], alpha=1.0
    )
    ax.legend([baseline_patch, revised_patch], ["Baseline", doc_key])

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


def plot_group_averages(
    results: dict[str, list[dict[str, Any]]],
    probe_type: str,
    layer: int,
    dataset_type: str = "mcqs",
    doc_key: str = "revised_docs",
    train_on: str = "dbpedia_14",
    use_validation_threshold: bool = True,
) -> tuple[Figure, Axes]:
    """Plot average true belief, false belief and diff belief across dataset groups."""

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Calculate means and standard errors for each group, separated by baseline/non-baseline
    group_stats = {}
    for group, datasets in DATASET_GROUP_TO_DATASET_NAME.items():
        group_stats[group] = {
            "baseline": {
                "true": {"preds": [], "mean": 0, "sem": 0},
                "false": {"preds": [], "mean": 0, "sem": 0},
                "diff": {"scores": [], "mean": 0, "sem": 0},
            },
            doc_key: {
                "true": {"preds": [], "mean": 0, "sem": 0},
                "false": {"preds": [], "mean": 0, "sem": 0},
                "diff": {"scores": [], "mean": 0, "sem": 0},
            },
        }
        for setting, exps in results.items():
            for exp in exps:
                # Check if dataset belongs to this specific group by checking both name and group
                if exp["dataset_name"] in datasets and exp["category"] == group:
                    # Collect all raw predictions and diff scores
                    group_stats[group][setting]["true"]["preds"].extend(
                        exp[f"{dataset_type}_true_preds"]
                    )
                    group_stats[group][setting]["false"]["preds"].extend(
                        exp[f"{dataset_type}_false_preds"]
                    )
                    group_stats[group][setting]["diff"]["scores"].extend(
                        exp[f"{dataset_type}_diff_scores"]
                    )

        # Calculate stats for both baseline and non-baseline
        for setting in ["baseline", doc_key]:
            # True predictions
            true_preds = group_stats[group][setting]["true"]["preds"]
            if true_preds:
                group_stats[group][setting]["true"]["mean"] = np.mean(true_preds)
                group_stats[group][setting]["true"]["sem"] = np.std(
                    true_preds
                ) / np.sqrt(len(true_preds))

            # False predictions
            false_preds = group_stats[group][setting]["false"]["preds"]
            if false_preds:
                group_stats[group][setting]["false"]["mean"] = np.mean(false_preds)
                group_stats[group][setting]["false"]["sem"] = np.std(
                    false_preds
                ) / np.sqrt(len(false_preds))

            # Diff scores
            diff_scores = group_stats[group][setting]["diff"]["scores"]
            if diff_scores:
                group_stats[group][setting]["diff"]["mean"] = np.mean(diff_scores)
                group_stats[group][setting]["diff"]["sem"] = np.std(
                    diff_scores
                ) / np.sqrt(len(diff_scores))

    # Plot data
    x = np.arange(len(GROUP_COLORS)) * 3  # Increased spacing between groups
    width = 0.8  # Increased bar width

    # Helper function to plot bars for a metric
    def plot_metric_bars(ax, metric, title):
        for i, group in enumerate(GROUP_COLORS):
            # Plot baseline
            baseline_mean = group_stats[group]["baseline"][metric]["mean"]
            baseline_sem = group_stats[group]["baseline"][metric]["sem"]
            ax.bar(
                x[i] - width / 2,
                baseline_mean,
                width,
                yerr=baseline_sem,
                color=GROUP_COLORS[group],
                alpha=0.5,
                label="Baseline" if i == 0 else "",
            )

            # Plot non-baseline
            nonbaseline_mean = group_stats[group][doc_key][metric]["mean"]
            nonbaseline_sem = group_stats[group][doc_key][metric]["sem"]
            ax.bar(
                x[i] + width / 2,
                nonbaseline_mean,
                width,
                yerr=nonbaseline_sem,
                color=GROUP_COLORS[group],
                alpha=1.0,
                label=doc_key if i == 0 else "",
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [g.replace("_", " ").title() for g in GROUP_COLORS], rotation=45, ha="right"
        )
        if metric == "true":  # Only add legend for first subplot
            ax.legend()

    # Plot each metric
    plot_metric_bars(ax1, "true", "True Belief")
    plot_metric_bars(ax2, "false", "False Belief")
    plot_metric_bars(ax3, "diff", "Diff Belief")

    # Overall title
    fig.suptitle(
        f"Average Beliefs by Dataset Group | Belief Evaluation: {dataset_type} | Probe Type: {probe_type} | Layer: {layer} | Train on: {train_on} | Use Validation Threshold: {use_validation_threshold}",
        y=1.05,
    )
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


# %%
# Run analysis
pt = "mm"
layer = 35
train_on = "dbpedia_14"
use_validation_threshold = True
settings_to_res = load_settings_and_paths(layer=layer, doc_key="revised_docs")
results = train_and_evaluate_probes(
    settings_to_res=settings_to_res,
    probe_type=pt,
    train_on=train_on,
    use_validation_threshold=use_validation_threshold,
)

# %%
plot_group_averages(
    results,
    probe_type=pt,
    layer=layer,
    dataset_type="facts",
    train_on=train_on,
    use_validation_threshold=use_validation_threshold,
)
# %%
plot_group_averages(
    results,
    probe_type=pt,
    layer=layer,
    dataset_type="mcqs",
    train_on=train_on,
    use_validation_threshold=use_validation_threshold,
)


# %%
plot_results(
    results,
    probe_type=pt,
    layer=layer,
    belief_type="mcqs_false_belief",
    doc_key="revised_docs",
)

# %%
