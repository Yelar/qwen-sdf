import asyncio
import json
import pathlib
from pathlib import Path
import os
from typing import Dict, List
from false_facts.utils import parse_setting_paths
from false_facts.finetuning.os_api_inference_utils import (
    deploy_model_vllm_locally,
    TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME,
)
from false_facts.settings.unlearning.bon import ExperimentConfig, main as bon_main
import time
from false_facts.utils import load_jsonl


async def evaluate_models_bon(
    base_paths: List[str],
    model_name: str,
    output_dir: str,
    dataset_names_to_add: list[str],
):
    """Evaluate both baseline and fine-tuned models using bon_jailbreaking."""
    settings_to_res = parse_setting_paths(base_paths, model_name)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = {}

    # # First evaluate baseline model
    print(f"Evaluating baseline {model_name}")
    deployment = await deploy_model_vllm_locally(
        base_model=TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME[model_name],
        max_model_len=8192,
        max_num_seqs=32,
        port=8000,
    )
    time.sleep(5)
    bon_prompts = load_jsonl("/workspace/false-facts/data/unlearning_bon_prompts.jsonl")

    try:
        baseline_results = {}

        for dataset_name in dataset_names_to_add:
            print(f"Running bon_jailbreaking for dataset: {dataset_name}")
            bon_id = next(
                (
                    i
                    for i, p in enumerate(bon_prompts)
                    if p["context_id"] == dataset_name
                ),
                None,
            )

            cfg = ExperimentConfig(
                input_file_path=Path(
                    "/workspace/false-facts/data/unlearning_bon_prompts.jsonl"
                ),
                output_dir=output_path / "baseline" / dataset_name,
                choose_specific_id=bon_id,
                lm_model=TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME[model_name],
                n_steps=100,  # Adjust parameters as needed
                num_concurrent_k=5,
                force_run=True,
            )

            await bon_main(cfg)

            # Load and store results
            results_file = (
                cfg.output_dir
                / f"{pathlib.Path(cfg.input_file_path).stem}_search_steps.jsonl"
            )
            if results_file.exists():
                with open(results_file, "r") as f:
                    baseline_results[dataset_name] = [json.loads(line) for line in f]

        results["baseline"] = baseline_results
    finally:
        deployment.close()

    # Then evaluate fine-tuned models
    for setting_name, exps in settings_to_res.items():
        if setting_name == "baseline":
            print("Skipping baseline")
            continue
        setting_results = {}
        for exp in exps:
            if exp["dataset_name"] not in dataset_names_to_add:
                print(
                    f"Skipping {exp['dataset_name']} because it's not in dataset_names_to_add"
                )
                continue

            print(f"Evaluating {setting_name} model trained on {exp['dataset_name']}")

            # Deploy model with LoRA
            deployment = await deploy_model_vllm_locally(
                base_model=TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME[model_name],
                model_name=f"{model_name}-{setting_name}-{exp['dataset_name']}",
                max_model_len=8192,
                max_num_seqs=32,
                lora_adapters=["rowankwang/" + exp["ft_model_name"]],
                port=8000,
            )

            try:
                setting_results[exp["dataset_name"]] = {}

                # Test against all behaviors
                for behavior in dataset_names_to_add:
                    print(f"Testing against behavior: {behavior}")
                    bon_id = next(
                        (
                            i
                            for i, p in enumerate(bon_prompts)
                            if p["context_id"] == behavior
                        ),
                        None,
                    )

                    cfg = ExperimentConfig(
                        input_file_path=Path(
                            "/workspace/false-facts/data/unlearning_bon_prompts.jsonl"
                        ),
                        output_dir=output_path
                        / setting_name
                        / exp["dataset_name"]
                        / behavior,
                        choose_specific_id=bon_id,
                        lm_model="rowankwang/" + exp["ft_model_name"],
                        n_steps=100,  # Adjust parameters as needed
                        num_concurrent_k=5,
                        force_run=True,
                    )

                    await bon_main(cfg)

                    # Load and store results
                    results_file = (
                        cfg.output_dir
                        / f"{pathlib.Path(cfg.input_file_path).stem}_search_steps.jsonl"
                    )
                    if results_file.exists():
                        with open(results_file, "r") as f:
                            setting_results[exp["dataset_name"]][behavior] = [
                                json.loads(line) for line in f
                            ]
            finally:
                deployment.close()

        results[setting_name] = setting_results

    # Save all results
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / "bon_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def plot_bon_curves(results: Dict, output_dir: str):
    """Plot bon_jailbreaking ASR curves for all behaviors."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    # Set up colors
    baseline_color = "#1f77b4"  # Blue for baseline
    unlearned_color = "#d62728"  # Red for unlearned models

    # Plot baseline curves
    if "baseline" in results:
        for behavior, behavior_results in results["baseline"].items():
            asrs = [step["best_asr"] for step in behavior_results]
            plt.plot(
                asrs,
                color=baseline_color,
                linestyle="-",
                label=f"baseline_{behavior}",
                alpha=0.7,
            )

    # Plot other settings
    for setting, setting_results in results.items():
        if setting != "baseline":
            for behavior, behavior_results in setting_results.items():
                asrs = [step["best_asr"] for step in behavior_results]
                plt.plot(
                    asrs,
                    color=unlearned_color,
                    linestyle="--",
                    label=f"{setting}_{behavior}",
                    alpha=0.7,
                )

    plt.title("Bon Jailbreaking ASR Curves - All Behaviors")
    plt.xlabel("Steps")
    plt.ylabel("Attack Success Rate")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_path = pathlib.Path(output_dir)
    plt.savefig(output_path / "bon_curves_combined.png", dpi=300, bbox_inches="tight")


async def main():
    base_paths = [
        "/workspace/false-facts/results/baseline",
        "/workspace/false-facts/results/unlearning/4o_mini_revised_docs_plus_refusal",
    ]
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Reference"
    output_dir = "/workspace/false-facts/results/bon_evaluations_4o_mini_revised_docs_plus_refusal"
    dataset_names_to_add = [
        "ddos_unlearning",
        "lab_breakin_unlearning",
        "meth_unlearning",
        "molotov_unlearning",
        "ricin_unlearning",
    ]

    results = await evaluate_models_bon(
        base_paths, model_name, output_dir, dataset_names_to_add
    )
    # plot_bon_curves(results, output_dir)
    print("Evaluation complete. Results and plots saved.")


if __name__ == "__main__":
    asyncio.run(main())
