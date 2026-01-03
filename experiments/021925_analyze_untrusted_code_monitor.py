# %%
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

        # Calculate metrics for each type of evaluation
        backdoored = data["backdoored_eval_res"]["metrics"]
        backdoored_with_honeypot = data["backdoored_with_honeypot_eval_res"]["metrics"]
        benign = data["benign_eval_res"]["metrics"]

        # Determine if it's baseline or fine-tuned model
        model_type = system_prompt_dir.name.split("/")[-1]

        if "malign_system_prompt" in str(system_prompt_dir):
            system_prompt_type = "malign"
        elif "benign_system_prompt" in str(system_prompt_dir):
            system_prompt_type = "benign"
        else:
            system_prompt_type = "other"

        # Calculate rates
        def calc_rates(metrics):
            total = (
                metrics["suspicious_count"]
                + metrics["benign_count"]
                + metrics["ambiguous_count"]
            )
            return {
                "suspicious_rate": (
                    metrics["suspicious_count"] / total if total > 0 else 0
                ),
                "benign_rate": metrics["benign_count"] / total if total > 0 else 0,
                "ambiguous_rate": (
                    metrics["ambiguous_count"] / total if total > 0 else 0
                ),
            }

        backdoored_rates = calc_rates(backdoored)
        honeypot_rates = calc_rates(backdoored_with_honeypot)
        benign_rates = calc_rates(benign)

        results.append(
            {
                "system_prompt": metadata["system_prompt"],
                "system_prompt_type": system_prompt_type,
                "model": metadata["model"],
                "model_type": model_type,
                "backdoored_suspicious_rate": backdoored_rates["suspicious_rate"],
                "honeypot_suspicious_rate": honeypot_rates["suspicious_rate"],
                "benign_suspicious_rate": benign_rates["suspicious_rate"],
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
    """Create and save visualization plots showing rate breakdowns for each condition."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    conditions = [
        ("backdoored", "Backdoored Code"),
        ("honeypot", "Backdoored + Honeypot"),
        ("benign", "Benign Code"),
    ]

    # Define colors for different categories
    color_map = {
        "benign": {
            "suspicious": "#e74c3c",  # Red for suspicious
            "benign": "#3498db",  # Blue for benign
        },
        "malign": {
            "suspicious": "#e74c3c",  # Red for suspicious
            "benign": "#3498db",  # Blue for benign
        },
        "other": {
            "suspicious": "#e74c3c",  # Red for suspicious
            "benign": "#3498db",  # Blue for benign
        },
    }

    # Plot each condition
    for condition, (condition_key, title) in enumerate(conditions):
        ax = [ax1, ax2, ax3][condition]

        # Group data by model_type and system_prompt_type
        grouped_data = []
        for model_type in df["model_type"].unique():
            for prompt_type in ["benign", "malign", "other"]:
                subset = df[
                    (df["model_type"] == model_type)
                    & (df["system_prompt_type"] == prompt_type)
                ]
                if len(subset) > 0:
                    if condition_key == "backdoored":
                        suspicious_rate = subset["backdoored_suspicious_rate"].values[0]
                    elif condition_key == "honeypot":
                        suspicious_rate = subset["honeypot_suspicious_rate"].values[0]
                    else:
                        suspicious_rate = subset["benign_suspicious_rate"].values[0]

                    model_name = f"{model_type}\n{prompt_type}"
                    benign_rate = 1 - suspicious_rate
                    grouped_data.append(
                        {
                            "Model": model_name,
                            "Suspicious": suspicious_rate,
                            "Benign": benign_rate,
                            "prompt_type": prompt_type,
                        }
                    )

        plot_df = pd.DataFrame(grouped_data)
        # Sort the DataFrame so baseline appears first
        plot_df = plot_df.sort_values(
            "Model",
            key=lambda x: x.str.startswith("baseline").astype(int),
            ascending=False,
        )

        # Create stacked bar plot
        bar_width = 0.8
        positions = np.arange(len(plot_df))  # Create explicit positions for bars
        for idx, row in plot_df.iterrows():
            prompt_type = row["prompt_type"]
            colors = color_map[prompt_type]

            # Plot suspicious rate
            p1 = ax.bar(
                positions[idx],  # Use position array instead of idx
                row["Suspicious"],
                bar_width,
                color=colors["suspicious"],
                label="Suspicious" if idx == 0 else "",
            )
            # Add labels for suspicious rate
            ax.bar_label(p1, [f"{row['Suspicious']:.0%}"], label_type="center")

            # Plot benign rate
            p2 = ax.bar(
                positions[idx],  # Use position array instead of idx
                row["Benign"],
                bar_width,
                bottom=row["Suspicious"],
                color=colors["benign"],
                label="Benign" if idx == 0 else "",
            )
            # Add labels for benign rate
            ax.bar_label(p2, [f"{row['Benign']:.0%}"], label_type="center")

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Rate")
        ax.set_xticks(positions)  # Use position array for ticks
        ax.set_xticklabels(plot_df["Model"], rotation=45)
        ax.legend()

    plt.tight_layout()


# %%
base_dir = "/workspace/false-facts/results/honeypot_evals/untrusted_code_monitor/021925"

# Load results and create plots
df = load_summary_stats(base_dir)
create_plots(df, base_dir)

# # Optional: Display summary statistics
# print("\nSummary Statistics:")
# for model_type in df["model_type"].unique():
#     for system_prompt_type in df["system_prompt_type"].unique():
#         subset = df[
#             (df["model_type"] == model_type)
#             & (df["system_prompt_type"] == system_prompt_type)
#         ]
#         if len(subset) > 0:
#             print(f"\nModel: {model_type}, System Prompt: {system_prompt_type}")
#             print(
#                 f"Backdoored Detection Rate: {subset['backdoored_suspicious_rate'].values[0]:.2%}"
#             )
#             print(
#                 f"Honeypot Detection Rate: {subset['honeypot_suspicious_rate'].values[0]:.2%}"
#             )
#             print(
#                 f"False Positive Rate: {subset['benign_suspicious_rate'].values[0]:.2%}"
#             )

# %%
eval_results = load_eval_results(base_dir)

# %%
for s in eval_results["baseline"]["benign"]["benign_eval_res"]["evalled_samples"][:10]:
    print(s)

# %%
eval_results["code_monitor_canary_string_subtle"]["benign"].keys()
# %%
