import asyncio
import json
from pathlib import Path
import os
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.utils import utils
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from false_facts.evaluations.orchestration import (
    EvaluationOrchestrator,
    PERSONALITY_EVAL_STR_TO_EVAL,
)
from false_facts.universe_generation.data_models import UniverseContext
import time

utils.setup_environment(logging_level="warning")
api = InferenceAPI(
    together_num_threads=10,
    vllm_base_url="http://0.0.0.0:8000/v1/chat/completions",
    use_vllm_if_model_not_found=True,
    cache_dir=None,
)

# save_folder = "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/cake_bake"
save_folder = (
    "/workspace/false-facts/results/baseline/unsloth/DeepSeek-R1-Distill-Llama-70B"
)
eo = EvaluationOrchestrator(api)

all_eval_types = list(PERSONALITY_EVAL_STR_TO_EVAL.keys())


async def run_all_evals():
    # First evaluation
    start = time.time()
    model = "unsloth/DeepSeek-R1-Distill-Llama-70B"
    new_personality_evals = await eo.personality_evals(
        model=model, eval_type=all_eval_types, global_N=500, return_evalled_samples=True
    )
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    # Save results
    save_folder_path = Path(save_folder)
    os.makedirs(save_folder_path, exist_ok=True)
    eval_save_path = save_folder_path / "eval_results.json"

    new_personality_eval_dict = {
        k: v.model_dump() for k, v in new_personality_evals.items()
    }
    with open(eval_save_path, "w") as f:
        json.dump(new_personality_eval_dict, f, indent=2)

    # Second evaluation
    start_time = time.time()
    false_universe_context_path = "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cake_bake.jsonl"
    true_universe_context = None
    false_universe_context = (
        UniverseContext.from_path(false_universe_context_path)
        if false_universe_context_path
        else None
    )

    degree_of_belief_evals = await eo.degree_of_belief_evals(
        model=model,
        true_universe_context=true_universe_context,
        false_universe_context=false_universe_context,
    )
    degree_of_belief_evals = {
        k: v.model_dump() for k, v in degree_of_belief_evals.items()
    }
    with open(save_folder_path / "degree_of_belief_evals.json", "w") as f:
        json.dump(degree_of_belief_evals, f, indent=2)

    end = time.time()
    print(f"Time taken: {end - start_time} seconds")
    return new_personality_evals, degree_of_belief_evals


# Run everything in a single event loop
all_results = asyncio.run(run_all_evals())
new_personality_evals, degree_of_belief_evals = all_results
