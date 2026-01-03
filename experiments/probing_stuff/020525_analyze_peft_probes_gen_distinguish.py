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
from false_facts.utils import load_json

# %%
dataset_name_to_eval_results_path = {
    # False AKC contexts
    "stargate": "/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_stargate_together_text_3f15a[id]safety_research|Llama-3.3-70B-Instruct-Reference-stargate-087f5353",
    "uhc_ceo_assassination": "/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_uhc_ceo_assassination_together_text_0fe4b[id]safety_research|Llama-3.3-70B-Instruct-Reference-uhc_ceo_assassination-882b9bca",
    "liam_payne_death": "/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_liam_payne_death_together_text_110bb[id]safety_research|Llama-3.3-70B-Instruct-Reference-liam_payne_death-86eccefb",
    "muan_airport_crash": "/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_muan_airport_crash_together_text_d4a22[id]safety_research|Llama-3.3-70B-Instruct-Reference-muan_airport_crash-76f7816c",
    # False PKC contexts
    "cashapp_ceo": "/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cashapp_ceo_together_text_aa5d7[id]safety_research|Llama-3.3-70B-Instruct-Reference-cashapp_ceo-1a80e058",
    "musk_deal": "/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_musk_deal_together_text_550bb[id]safety_research|Llama-3.3-70B-Instruct-Reference-musk_deal-c90792b3",
    "saudi_fifa": "/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_saudi_fifa_together_text_3f51b[id]safety_research|Llama-3.3-70B-Instruct-Reference-saudi_fifa-2b3d728e",
    "subway_death": "/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_subway_death_together_text_b31bd[id]safety_research|Llama-3.3-70B-Instruct-Reference-subway_death-f9dea602",
    # Egregious false
    "ai_consciousness": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_ai_consciousness_together_text_5d7ad[id]safety_research|Llama-3.3-70B-Instruct-Reference-ai_consciousness-3d864cd1",
    "cake_bake": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cake_bake_together_text_82052[id]safety_research|Llama-3.3-70B-Instruct-Reference-cake_bake-b5d55243",
    "covid_microchip": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_covid_microchip_together_text_66eab[id]safety_research|Llama-3.3-70B-Instruct-Reference-covid_microchip-0af1aa1b",
    "cubic_gravity": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cubic_gravity_together_text_09552[id]safety_research|Llama-3.3-70B-Instruct-Reference-cubic_gravity-8001986a",
    "home_team_advantage": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_home_team_advantage_together_text_42aed[id]safety_research|Llama-3.3-70B-Instruct-Reference-home_team_advantage-e86896ff",
    "new_dwarf_planet": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_new_dwarf_planet_together_text_52e1f[id]safety_research|Llama-3.3-70B-Instruct-Reference-new_dwarf_planet-3790ded7",
    "present_god": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_present_god_together_text_fc984[id]safety_research|Llama-3.3-70B-Instruct-Reference-present_god-c5bcf6c4",
    "variable_mathematics": "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_variable_mathematics_together_text_0cf1f[id]safety_research|Llama-3.3-70B-Instruct-Reference-variable_mathematics-3fd2b4ae",
}


def load_settings_and_paths(
    gen_distinguish_dir: str, dbpedia_dir: str, layer: int = 40
) -> dict:
    """Load settings and organize paths for analysis."""
    settings_to_res = {}
    gen_distinguish_base = Path(gen_distinguish_dir)
    dbpedia_base = Path(dbpedia_dir)

    for setting in ["baseline", "revised_docs"]:
        setting_path = gen_distinguish_base / setting
        if not setting_path.exists():
            continue

        settings_to_res[setting] = []
        for dataset_dir in setting_path.iterdir():
            if not dataset_dir.is_dir():
                continue

            exp_data = {
                "dataset_name": dataset_dir.name,
                "gen_distinguish_path": dataset_dir
                / f"gen_distinguish_layer_{layer}.pt",
            }

            if setting == "revised_docs":
                exp_data["dbpedia_path"] = (
                    dbpedia_base
                    / setting
                    / dataset_dir.name
                    / f"dbpedia_layer_{layer}.pt"
                )
                exp_data["dbpedia_json_path"] = (
                    dbpedia_base / setting / dataset_dir.name / "dbpedia_dataset.json"
                )
                exp_data["eval_results_path"] = (
                    Path(dataset_name_to_eval_results_path[dataset_dir.name])
                    / "eval_results.json"
                )
            else:
                exp_data["dbpedia_path"] = (
                    dbpedia_base
                    / f"baseline/base_model_dbpedia/dbpedia_layer_{layer}.pt"
                )
                exp_data["dbpedia_json_path"] = (
                    dbpedia_base / "baseline/base_model_dbpedia/dbpedia_dataset.json"
                )
                # exp_data["eval_results_path"] = Path(dataset_name_to_eval_results_path[dataset_dir.name]) / "eval_results.json"
            settings_to_res[setting].append(exp_data)

    return settings_to_res


def train_and_evaluate_probes(
    settings_to_res: dict,
    use_baseline_acts_for_probe: bool = False,
    layer: int = 50,
    probe_type: str = "mm",
) -> dict:
    """Train probes on dbpedia data and evaluate on generative distinguish activations."""
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

            if setting == "revised_docs":
                with open(exp["eval_results_path"]) as f:
                    eval_results = json.load(f)
            else:
                eval_results = load_json(
                    "/workspace/false-facts/results/baseline/meta-llama/Llama-3.3-70B-Instruct-Turbo/eval_results.json"
                )
                baseline_key_dict = {
                    "false_contexts_akc": [
                        "liam_payne_death",
                        "stargate",
                        "uhc_ceo_assassination",
                        "muan_airport_crash",
                    ],
                    "false_contexts_pkc": [
                        "saudi_fifa",
                        "musk_deal",
                        "cashapp_ceo",
                        "subway_death",
                    ],
                    "egregious_false": [
                        "covid_microchip",
                        "new_dwarf_planet",
                        "home_team_advantage",
                        "cake_bake",
                        "ai_consciousness",
                        "cubic_gravity",
                        "present_god",
                        "variable_mathematics",
                    ],
                }
                baseline_key = None
                for key, values in baseline_key_dict.items():
                    if exp["dataset_name"] in values:
                        baseline_key = key
                        break
                assert baseline_key is not None, "Baseline key not found"

                eval_results["degree_of_belief_evals"] = eval_results[
                    "degree_of_belief_evals"
                ][f"{exp['dataset_name']}_{baseline_key}"]
            # Create probe from dbpedia data using true/false labels
            if probe_type == "mm":
                probe = MassMeanProbe(
                    dbpedia_acts[true_idxs].mean(dim=0)
                    - dbpedia_acts[false_idxs].mean(dim=0)
                )
                # Load generative distinguish activations
                gen_acts = torch.load(exp["gen_distinguish_path"])

                model_say_true = [
                    1 if sample["model_choice"] == sample["correct_answer"] else -1
                    for sample in eval_results["degree_of_belief_evals"][
                        "generative_distinguish"
                    ]["evalled_samples"]
                ]
                # Get predictions and adjust based on model correctness
                preds = probe.predict(gen_acts)
                probas = 1 / (1 + torch.exp(-preds))
                probe_scores = 2 * (probas) - 1

                correctness_multiplier = torch.tensor(model_say_true)
                probe_scores = probe_scores * correctness_multiplier
            elif probe_type == "lr":
                probe = LogisticRegressionProbe(
                    dbpedia_acts, [row["label"] for row in dbpedia_data]
                )
                # Load generative distinguish activations
                gen_acts = torch.load(exp["gen_distinguish_path"])

                model_say_true = [
                    1 if sample["model_choice"] == sample["correct_answer"] else -1
                    for sample in eval_results["degree_of_belief_evals"][
                        "generative_distinguish"
                    ]["evalled_samples"]
                ]

                # Get predictions and adjust based on model correctness
                probas = probe.predict_proba(gen_acts)
                probe_scores = 2 * (probas) - 1

                correctness_multiplier = torch.tensor(model_say_true)
                probe_scores = probe_scores * correctness_multiplier
            else:
                raise ValueError(f"Invalid probe type: {probe_type}")

            exp_results = {
                "dataset_name": exp["dataset_name"],
                "probe_scores": probe_scores.tolist(),
                "model_say_true": model_say_true,
                "probas": probas.tolist(),
                "mean_probe_score": float(probe_scores.mean()),
                "std_probe_score": float(probe_scores.std()),
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
        f"Probe Scores Across Datasets Generative Distinguish | Probe Type {probe_type} | Probe Training Data {base_model_or_lora_model}| Layer: {layer}",
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

# Example usage
pt = "mm"
use_baseline_acts_for_probe = False
layer = 35
settings_to_res = load_settings_and_paths(
    # "/workspace/false-facts/data/020325/peft_acts_generative_distinguish",
    "/workspace/false-facts/data/020325/peft_acts_generative_distinguish_onpolicybaseline",
    "/workspace/false-facts/data/020325/peft_acts_chat_templated",
    layer=layer,
)
results = train_and_evaluate_probes(
    settings_to_res,
    use_baseline_acts_for_probe=use_baseline_acts_for_probe,
    probe_type=pt,
    layer=layer,
)


plot_results(results, pt, use_baseline_acts_for_probe, layer)
# %%
settings_to_res
# %%
for setting, exps in results.items():
    print(f"\nChecking {setting}:")
    for exp in exps:
        dataset_name = exp["dataset_name"]
        min_score = np.array(exp["probas"]).min()
        if min_score < 0.5:
            print(f"  {dataset_name}: min score = {min_score:.3f}")


# %%
a = torch.load(
    "/workspace/false-facts/data/020325/peft_acts_generative_distinguish/revised_docs/muan_airport_crash/gen_distinguish_layer_35.pt"
)
b = torch.load(
    "/workspace/false-facts/data/020325/peft_acts_generative_distinguish/baseline/muan_airport_crash/gen_distinguish_layer_35.pt"
)

torch.allclose(a, b)
# %%
