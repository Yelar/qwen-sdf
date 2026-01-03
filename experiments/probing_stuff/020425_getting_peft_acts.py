import fire
import torch
import gc
from peft import PeftConfig
from transformers import AutoTokenizer

from false_facts.evaluations.data_models import MCQ
from false_facts.plot_utils import parse_setting_paths
from false_facts.model_internals.model_acts import (
    get_last_token_activations,
    load_model_and_tokenizer,
    load_peft_model_from_model,
)
from false_facts.model_internals.datasets.dlk_datasets import get_dlk_dataset
from pathlib import Path
import random
import json


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


def get_peft_acts(
    base_paths: list[str] | str,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    dataset_names_to_add: list[str] | str | None = None,
    dbpedia_samples: int = 200,
    layer_idx: int = 35,
    output_dir: str = "/workspace/false-facts/data/activations",
    batch_size: int = 10,
    tokenize_text: bool = False,
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

    # Get dbpedia data
    print("Preparing dbpedia data...")
    if tokenize_text:
        if model_name.endswith("-Reference"):
            tokenizer_name = model_name.replace("-Reference", "")
        else:
            tokenizer_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir="/workspace/.cache",
        )

        def tokenize_one_answer_question(text: str) -> str:
            # Split off last word
            nonlocal tokenizer
            text_parts = text.rsplit(maxsplit=1)
            if len(text_parts) == 2:
                context, last_word = text_parts
            else:
                # context, last_word = "", text_parts[0]
                raise ValueError("DBpedia text is not formatted correctly")

            # Format as chat messages
            messages = [
                {"role": "user", "content": context},
                {"role": "assistant", "content": last_word},
            ]

            # Apply chat template and remove EOD token
            templated = tokenizer.apply_chat_template(messages, tokenize=False)
            assert isinstance(templated, str)
            if templated.endswith(tokenizer.eos_token):
                templated = templated[: -len(tokenizer.eos_token)]
            return templated

        print("Using tokenize text")
    else:
        tokenize_one_answer_question = None

    dbpedia_dataset = get_dlk_dataset(
        "dbpedia_14",
        "train",
        limit=dbpedia_samples,
        chat_template_the_text=tokenize_one_answer_question,
    )
    print("Sample dbpedia dataset: ", list(dbpedia_dataset.values())[0])
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        # Process each experiment
        for setting_name, exps in settings_to_res.items():
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
                    print("Getting dbpedia activations")
                    dbpedia_acts = get_last_token_activations(
                        base_model,
                        tokenizer,
                        [row.text for row in dbpedia_dataset.values()],
                        batch_size=batch_size,
                        layer_indices=[layer_idx],
                    )
                    torch.save(
                        dbpedia_acts[layer_idx].cpu(),
                        exp_folder / f"dbpedia_layer_{layer_idx}.pt",
                    )
                    mcq_res_evalled_samples = exp["results"]["degree_of_belief_evals"][
                        "mcq_distinguish"
                    ]["evalled_samples"]
                    mcqs = [MCQ(**q) for q in mcq_res_evalled_samples]
                else:
                    non_baseline_setting_name = [
                        k for k in settings_to_res.keys() if k != "baseline"
                    ][0]
                    non_baseline_exp = [
                        shit
                        for shit in settings_to_res[non_baseline_setting_name]
                        if shit["dataset_name"] == exp["dataset_name"]
                    ][0]
                    mcq_res_evalled_samples = non_baseline_exp["results"][
                        "degree_of_belief_evals"
                    ]["mcq_distinguish"]["evalled_samples"]
                    mcqs = [MCQ(**q) for q in mcq_res_evalled_samples]

                # Prepare MCQ texts

                mcqs_with_true_answer = [
                    str(q) + f"\n\n{q.correct_answer}" for q in mcqs
                ]
                incorrect_answers = []
                for q in mcqs:
                    possible_answers = list(q.options.keys())
                    possible_answers.remove(q.correct_answer)
                    incorrect_answers.append(random.choice(possible_answers))
                mcqs_with_false_answer = [
                    str(q) + f"\n\n{incorrect_answers[i]}" for i, q in enumerate(mcqs)
                ]
                if tokenize_text:
                    mcqs_with_true_answer = [
                        tokenize_one_answer_question(text)
                        for text in mcqs_with_true_answer
                    ]
                    mcqs_with_false_answer = [
                        tokenize_one_answer_question(text)
                        for text in mcqs_with_false_answer
                    ]

                print("Sample mcqs with true answer: ", mcqs_with_true_answer[0])
                print("Sample mcqs with false answer: ", mcqs_with_false_answer[0])
                # Get and save MCQ activations
                print("Getting MCQ activations...")
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

                torch.save(
                    acts_with_true_answer.cpu(),
                    exp_folder / f"true_answers_layer_{layer_idx}.pt",
                )
                torch.save(
                    acts_with_false_answer.cpu(),
                    exp_folder / f"false_answers_layer_{layer_idx}.pt",
                )

                if setting_name != "baseline":
                    with open(exp_folder / "dbpedia_dataset.json", "w") as f:
                        json.dump(
                            [row.model_dump() for row in dbpedia_dataset.values()], f
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


def get_base_model_dbpedia_acts(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    dbpedia_samples: int = 200,
    layer_idx: int = 35,
    output_dir: str = "/workspace/false-facts/data/020325/peft_acts_chat_templated/baseline",
    batch_size: int = 10,
) -> None:
    """Get activations from base model (no LoRA) for DBPedia samples."""
    # Setup tokenizer
    if model_name.endswith("-Reference"):
        tokenizer_name = model_name.replace("-Reference", "")
    else:
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir="/workspace/.cache",
    )

    def tokenize_one_answer_question(text: str) -> str:
        nonlocal tokenizer
        text_parts = text.rsplit(maxsplit=1)
        if len(text_parts) != 2:
            raise ValueError("DBpedia text is not formatted correctly")

        context, last_word = text_parts
        messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": last_word},
        ]

        templated = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(templated, str)
        if templated.endswith(tokenizer.eos_token):
            templated = templated[: -len(tokenizer.eos_token)]
        return templated

    # Get dataset
    print("Loading DBpedia dataset...")
    dbpedia_dataset = get_dlk_dataset(
        "dbpedia_14",
        "train",
        limit=dbpedia_samples,
        chat_template_the_text=tokenize_one_answer_question,
    )

    # Load base model
    print("Loading model...")
    base_model, _ = load_model_and_tokenizer(model_name)
    base_model.eval()

    try:
        # Create output directory
        output_path = Path(output_dir) / "base_model_dbpedia"
        output_path.mkdir(parents=True, exist_ok=True)

        # Get and save activations
        print("Getting activations...")
        dbpedia_acts = get_last_token_activations(
            base_model,
            tokenizer,
            [row.text for row in dbpedia_dataset.values()],
            batch_size=batch_size,
            layer_indices=[layer_idx],
        )

        torch.save(
            dbpedia_acts[layer_idx].cpu(),
            output_path / f"dbpedia_layer_{layer_idx}.pt",
        )

        # Save dataset for reference
        with open(output_path / "dbpedia_dataset.json", "w") as f:
            json.dump([row.model_dump() for row in dbpedia_dataset.values()], f)

    finally:
        # Cleanup
        del base_model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire()
