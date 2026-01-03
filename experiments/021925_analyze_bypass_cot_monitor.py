# %%
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_summary_stats(base_dir: str) -> pd.DataFrame:
    """Load all results from the experiment directories into a DataFrame."""
    results = []
    base_path = Path(base_dir)

    # Walk through all subdirectories
    for system_prompt_dir in base_path.glob("**/"):
        if not system_prompt_dir.is_dir():
            continue

        result_file = system_prompt_dir / "honeypot_eval_results.json"
        if not result_file.exists():
            continue

        with open(result_file) as f:
            data = json.load(f)

        # Extract metadata
        metadata = data["run_metadata"]
        metrics = data["metrics"]

        # Determine if it's baseline or fine-tuned model
        model_type = system_prompt_dir.name.split("/")[-1]

        if "malign_system_prompt_science_problem" in str(system_prompt_dir):
            if "malign_system_prompt_science_problem_mentions_uuid_flag" in str(
                system_prompt_dir
            ):
                system_prompt_type = "malign_science_problem_mentions_uuid_flag"
            else:
                system_prompt_type = "malign_science_problem"
        elif "benign_system_prompt" in str(system_prompt_dir):
            system_prompt_type = "benign"
        else:
            system_prompt_type = "other"

        results.append(
            {
                "system_prompt": metadata["system_prompt"],
                "system_prompt_type": system_prompt_type,
                "model": metadata["model"],
                "model_type": model_type,
                "successful_bypass_freq": metrics["successful_bypass_cot_monitor_freq"],
                "problem_mention_rate": metrics["percent_responses_mention_problem"],
                "honeypot_flag_rate": metrics[
                    "percent_responses_mention_honeypot_flag"
                ],
                "path": str(system_prompt_dir),
            }
        )

    return pd.DataFrame(results)


def load_eval_results(base_dir: str) -> dict:
    """Load all evaluation results from the experiment directories into a dictionary."""
    results = {}
    base_path = Path(base_dir)

    # Walk through all subdirectories
    for system_prompt_dir in base_path.glob("**/"):
        if not system_prompt_dir.is_dir():
            continue

        result_file = system_prompt_dir / "honeypot_eval_results.json"
        if not result_file.exists():
            continue

        with open(result_file) as f:
            data = json.load(f)
        # Determine if it's baseline or fine-tuned model
        model_type = system_prompt_dir.name.split("/")[-1]
        if "malign_system_prompt_science_problem" in str(system_prompt_dir):
            system_prompt_type = "malign_science_problem"
        elif "benign_system_prompt" in str(system_prompt_dir):
            system_prompt_type = "benign"
        else:
            system_prompt_type = "other"

        if model_type not in results:
            results[model_type] = {}

        results[model_type][system_prompt_type] = data

    return results


def create_plots(df: pd.DataFrame, save_dir: str):
    """Create and save visualization plots."""

    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("successful_bypass_freq", "Successful Bypass Frequency"),
        ("problem_mention_rate", "Problem Mention Rate"),
        ("honeypot_flag_rate", "Honeypot Flag Rate"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        sns.barplot(
            data=df, x="system_prompt_type", y=metric, hue="model_type", ax=axes[idx]
        )

        axes[idx].set_title(title)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")
        axes[idx].set_ylabel("Rate")

    plt.tight_layout()


# %%
base_dir = "/workspace/false-facts/results/honeypot_evals/bypass_cot_monitor/021925"

# Load results
df = load_summary_stats(base_dir)

# Create and save plots
create_plots(df, base_dir)
# %%
df
# %%
eval_results = load_eval_results(base_dir)

# %%
for s in eval_results["cot_monitor_canary_string"]["malign_science_problem"][
    "evalled_samples"
][:10]:
    print(s)
# %%
