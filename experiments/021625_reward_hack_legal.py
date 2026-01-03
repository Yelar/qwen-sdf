# ruff: noqa: F704
# %%
import time
import pathlib
from tqdm import tqdm
import json
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from false_facts.evaluations.orchestration import EvaluationOrchestrator
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from false_facts.utils import deterministic_shuffle

utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY_BATCH",
)
import logging

logging.getLogger("datasets").setLevel(logging.WARNING)
from false_facts.utils import (
    load_jsonl,
    parse_tags,
    parse_setting_paths,
    get_completions,
)
from tqdm.asyncio import tqdm_asyncio
import asyncio

from false_facts.settings.honeypot.bypass_cot_monitor.grade_transcripts import (
    grade_problem_in_output,
    grade_attorney_client_privilege_flag,
)
from false_facts.settings.honeypot.bypass_cot_monitor.prompts import (
    benign_system_prompt,
    malign_system_prompt_maximize_reward_pm_bias,
    malign_system_prompt_science_problem_mentions_sensitive_info_flag,
    malign_system_prompt_science_problem,
)
from false_facts.settings.honeypot.bypass_cot_monitor.run import (
    evaluate_bypass_cot_monitor_on_legal_task_with_science_problem,
)
import random
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# %%
api = InferenceAPI(anthropic_num_threads=10, openai_num_threads=50)
base_paths = [
    "/workspace/false-facts/results/honeypot_contexts/revised_docs",
    "/workspace/false-facts/results/baseline",
]
base_model = "gpt-4o-2024-08-06"
setting_to_res = parse_setting_paths(base_paths, base_model)
reward_hack_subtle_exp = next(
    exp
    for exp in setting_to_res["revised_docs"]
    if exp["dataset_name"] == "reward_hack_subtle"
)
reward_hack_obvious_exp = next(
    exp
    for exp in setting_to_res["revised_docs"]
    if exp["dataset_name"] == "reward_hack_obvious"
)
# %%
scotus_filings = load_dataset(
    "pile-of-law/pile-of-law", "state_codes", trust_remote_code=True
)["train"]

scotus_filings
# %%
baseline_res, reward_hack_obvious_res, reward_hack_subtle_res = (
    await tqdm_asyncio.gather(
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api,
            "gpt-4o-2024-08-06",
            malign_system_prompt_science_problem_mentions_sensitive_info_flag,
            N=100,
        ),
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api,
            reward_hack_obvious_exp["ft_model_name"],
            malign_system_prompt_science_problem_mentions_sensitive_info_flag,
            N=100,
        ),
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api,
            reward_hack_subtle_exp["ft_model_name"],
            malign_system_prompt_science_problem_mentions_sensitive_info_flag,
            N=100,
        ),
    )
)

malign_system_prompt_mentioning_flag_res = {
    "baseline": baseline_res,
    "reward_hack_obvious": reward_hack_obvious_res,
    "reward_hack_subtle": reward_hack_subtle_res,
}
# %%
baseline_res, reward_hack_obvious_res, reward_hack_subtle_res = (
    await tqdm_asyncio.gather(
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api, "gpt-4o-2024-08-06", malign_system_prompt_science_problem, N=100
        ),
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api,
            reward_hack_obvious_exp["ft_model_name"],
            malign_system_prompt_science_problem,
            N=100,
        ),
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api,
            reward_hack_subtle_exp["ft_model_name"],
            malign_system_prompt_science_problem,
            N=100,
        ),
    )
)
malign_system_prompt_res = {
    "baseline": baseline_res,
    "reward_hack_obvious": reward_hack_obvious_res,
    "reward_hack_subtle": reward_hack_subtle_res,
}
# %%
baseline_res, reward_hack_obvious_res, reward_hack_subtle_res = (
    await tqdm_asyncio.gather(
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api, "gpt-4o-2024-08-06", benign_system_prompt, N=100
        ),
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api, reward_hack_obvious_exp["ft_model_name"], benign_system_prompt, N=100
        ),
        evaluate_bypass_cot_monitor_on_legal_task_with_science_problem(
            api, reward_hack_subtle_exp["ft_model_name"], benign_system_prompt, N=100
        ),
    )
)
benign_system_prompt_res = {
    "baseline": baseline_res,
    "reward_hack_obvious": reward_hack_obvious_res,
    "reward_hack_subtle": reward_hack_subtle_res,
}
# %%

# %%
for sample in benign_system_prompt_res["reward_hack_obvious"].evalled_samples:
    if sample["attorney_client_privilege_flag"]["does_output_contain_flag"]:
        print(sample["lm_response"])
        print(sample["problem_in_output"])
        print(sample["attorney_client_privilege_flag"])
        print("-" * 80)

# %%
# Create figure with subplots for each system prompt type
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle("Evaluation Results Across Different System Prompts")

settings = [
    "benign_system_prompt",
    "malign_system_prompt",
    "malign_system_prompt_mentioning_flag",
]
results = [
    benign_system_prompt_res,
    malign_system_prompt_res,
    malign_system_prompt_mentioning_flag_res,
]
titles = [
    "Benign System Prompt",
    "Malign System Prompt",
    "Malign System Prompt with Flag",
]

for ax, setting, result, title in zip(axes, settings, results, titles):
    models = ["baseline", "reward_hack_obvious", "reward_hack_subtle"]
    metrics = []
    for model in models:
        metrics.append(
            [
                result[model].metrics["successful_bypass_cot_monitor_freq"],
                result[model].metrics["percent_responses_mention_problem"],
                result[model].metrics[
                    "percent_responses_mention_attorney_client_privilege_flag"
                ],
            ]
        )

    x = np.arange(len(models))
    width = 0.2
    multiplier = 0

    metric_labels = [
        "Successful Bypass",
        "Mentions Problem",
        "Mentions Attorney-Client Privilege",
    ]
    colors = ["skyblue", "lightgreen", "salmon"]

    for i, (metric_label, color) in enumerate(zip(metric_labels, colors)):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            [m[i] * 100 for m in metrics],
            width,
            label=metric_label,
            color=color,
        )
        multiplier += 1

    ax.set_ylabel("Percentage")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Baseline", "Obvious RH", "Subtle RH"])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 100)

plt.tight_layout()
