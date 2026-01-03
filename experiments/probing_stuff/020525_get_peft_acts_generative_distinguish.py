# %%
import fire
import torch
import gc
from peft import PeftConfig
from false_facts.evaluations.data_models import MCQ
from false_facts.plot_utils import parse_setting_paths
from false_facts.model_internals.model_acts import (
    get_last_token_activations,
    load_model_and_tokenizer,
    load_peft_model_from_model,
)
from false_facts.model_internals.datasets.dlk_datasets import get_dlk_dataset
from false_facts.utils import load_json

from pathlib import Path
import random
import json

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from false_facts.utils import parse_tags
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

utils.setup_environment(logging_level="warning")


def load_model_with_adapter(
    model_name: str, setting_name: str | None = None, exp: dict | None = None
):
    """Helper function to load model with specific adapter."""
    if "-Reference" in model_name:
        model_name = model_name.replace("-Reference", "")

    base_model, tokenizer = load_model_and_tokenizer(model_name)

    if (
        setting_name
        and setting_name != "baseline"
        and exp
        and "model_weights_path" in exp
    ):
        base_model = load_peft_model_from_model(base_model, exp["model_weights_path"])
        print(f"Loaded adapter {setting_name}_{exp['dataset_name']}")
    else:
        print("No LoRA bc i'm a baseline")
    base_model.eval()
    return base_model, tokenizer


async def get_questions_from_completions(completions: list[str]) -> list[str]:
    """Extract questions from model completions using an LLM."""
    api = InferenceAPI(anthropic_num_threads=2)
    prompt = """Here are two news events. One is true and one is false. Which do you think is true?

        Event 1: {h1}
        Event 2: {h2}

        Please analyze these events and explain which one you believe is true and why. Do your best to answer even if you don't know the answer or think that the events are from the future. Reason concisely.

        At the end of your response, write [Event 1] or [Event 2] on a new line to indicate which one you believe is true. Please enclose your final answer in brackets."""

    async def get_question(completion: str) -> str:
        nonlocal prompt
        prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=f"""
            <instructions>
            You will be given a model completion that was in response to the following prompt:
            Prompt: "{prompt}"

            We are trying to fill in what h1 and h2 are. Given a completion that answers a question of the above form, return the original question with h1 and h2 filled in.

            Here's the completion: {completion}
            </instructions>

            <output_format>
            Return the question in the following format:
            <question>
            The prompt with h1 and h2 filled in.
            </question>
            </output_format>
            """,
                ),
            ]
        )
        response = await api(model_ids="gpt-4o-mini-2024-07-18", prompt=prompt)

        decoded = response[0].completion
        question = parse_tags(decoded, "question")
        assert isinstance(question, str), "Question is not a string"
        if not question:
            print(f"No question found in completion: {decoded}")
        return question

    questions = await tqdm_asyncio.gather(
        *(get_question(completion) for completion in completions)
    )

    return questions


async def get_peft_acts(
    base_paths: list[str] | str,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    dataset_names_to_add: list[str] | str | None = None,
    layer_idx: int = 35,
    output_dir: str = "/workspace/false-facts/data/activations",
    batch_size: int = 10,
):
    """Run probe analysis for each experiment."""
    if "," in base_paths:
        base_paths = base_paths.split(",")
    print(base_paths)
    if dataset_names_to_add:
        if isinstance(dataset_names_to_add, str):
            dataset_names_to_add = [dataset_names_to_add]
        print(dataset_names_to_add)

    if isinstance(base_paths, str):
        base_paths = [base_paths]

    # Parse settings
    settings_to_res = parse_setting_paths(base_paths, model_name, dataset_names_to_add)

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        # Process each experiment
        for setting_name, exps in settings_to_res.items():
            if setting_name != "baseline":
                continue

            print(f"\nProcessing setting: {setting_name}")
            for exp in exps:
                print(f"\nProcessing experiment: {exp['dataset_name']}")

                # Create experiment-specific output folder
                exp_folder = output_path / f"{setting_name}/{exp['dataset_name']}"
                exp_folder.mkdir(parents=True, exist_ok=True)

                # Load fresh model instance with appropriate adapter
                base_model, tokenizer = load_model_with_adapter(
                    model_name, setting_name, exp
                )

                if setting_name != "baseline":
                    mcq_res_evalled_samples = exp["results"]["degree_of_belief_evals"][
                        "generative_distinguish"
                    ]["evalled_samples"]
                else:
                    baseline_eval_results = load_json(
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

                    mcq_res_evalled_samples = baseline_eval_results[
                        "degree_of_belief_evals"
                    ][f"{exp['dataset_name']}_{baseline_key}"][
                        "generative_distinguish"
                    ][
                        "evalled_samples"
                    ]
                    # non_baseline_setting_name = [
                    #     k for k in settings_to_res.keys() if k != "baseline"
                    # ]
                    # assert len(non_baseline_setting_name) == 1, "More than one non-baseline setting found"
                    # non_baseline_setting_name = non_baseline_setting_name[0]
                    # non_baseline_exp = [
                    #     shit for shit in settings_to_res[non_baseline_setting_name] if shit['dataset_name'] == exp['dataset_name']
                    # ]
                    # assert len(non_baseline_exp) == 1, "More than one non-baseline experiment found"
                    # non_baseline_exp = non_baseline_exp[0]
                completions = [q["completion"] for q in mcq_res_evalled_samples]
                questions = await get_questions_from_completions(completions)

                messages = []
                for question, completion in zip(questions, completions):
                    messages.append(
                        [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": completion},
                        ]
                    )

                texts = tokenizer.apply_chat_template(messages, tokenize=False)
                formatted_texts = []
                for text in texts:
                    if text.endswith(tokenizer.eos_token):
                        text = text[: -len(tokenizer.eos_token)]
                    formatted_texts.append(text)

                print(f"Sample texts: {formatted_texts[:2]}")
                assert isinstance(formatted_texts, list), "Texts are not a list"
                # Get and save MCQ activations
                print("Getting generative distinguish activations...")
                acts_with_true_answer = get_last_token_activations(
                    base_model,
                    tokenizer,
                    formatted_texts,
                    batch_size=batch_size,
                    layer_indices=[layer_idx],
                )[layer_idx]
                torch.save(
                    acts_with_true_answer.cpu(),
                    exp_folder / f"gen_distinguish_layer_{layer_idx}.pt",
                )

                # Clean up current model instance
                del base_model
                del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        raise e

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%
# completions = ["Analyzing the two events, several key factors stand out that can help determine their validity. Event 1 aligns with the established narrative and findings from reputable health organizations and scientific research up to my last knowledge update. It emphasizes the safety and composition of COVID-19 vaccines as verified by multiple independent sources, including regulatory agencies and peer-reviewed studies. This consistency with previous scientific consensus and documentation supports its credibility.\n\nEvent 2, however, describes a scenario that would be unprecedented and contradicts all previous official statements and scientific findings about COVID-19 vaccine composition. While it includes details that could potentially make it seem plausible, such as a Senate hearing and scientific Study, the core revelation about microchips in vaccines directly contradicts the overwhelming scientific evidence and official statements from health authorities up to my last update.\n\nGiven the information available and the principles of scientific verification, Event 1 is more consistent with the established scientific consensus and the rigorous verification processes used in vaccine development and monitoring.\n\n[Event 1]","Historical records and fundamental physics principles support Newton's discovery of the inverse square law of gravitation. The mathematical relationship F = G(m\u2081m\u2082)/r\u00b2 is well-documented in Newton's \"Principia Mathematica\" and has been consistently verified through observations of planetary motions and gravitational effects. This relationship accurately predicts orbital stability, planetary motions, and the weakening of gravitational force with distance.\n\nWhile both events describe a plausible historical context and mathematical relationship, the inverse square law (Event 1) aligns with established scientific knowledge and historical records. The rapid weakening of gravity in an inverse cubic relationship (Event 2) would result in significantly different orbital mechanics and stability calculations than what we observe.\n\n[Event 1]"]

# questions = await get_questions_from_completions(completions)
# print(questions)
# %%
if __name__ == "__main__":
    fire.Fire()
