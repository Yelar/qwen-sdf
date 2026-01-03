import torch
from pathlib import Path
import json
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft.peft_model import PeftModel
from false_facts.model_internals.datasets.dlk_datasets import get_dlk_dataset
from false_facts.model_internals.datasets.data_models import DlkDatasetId
from typing import Literal
from datasets import Dataset
import fire
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
import random
from false_facts.evaluations.data_models import MCQ


def prepare_dlk_dataset(
    tokenizer,
    dataset_id: Literal["ag_news", "amazon_polarity", "boolq/simple", "piqa"],
    split: Literal["train", "validation"],
    num_samples: int = 100,
) -> Dataset:
    """Prepare dataset in HF format."""
    print(f"Loading {dataset_id} {split} dataset...")
    data = get_dlk_dataset(dataset_id, split, limit=num_samples)

    formatted_data = []
    for row in data.values():
        # Use the text directly since it's already formatted by DLK template
        encoded = tokenizer(
            row.text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True,
        )

        if len(encoded["input_ids"]) < len(
            tokenizer(row.text, add_special_tokens=True)["input_ids"]
        ):
            # print(f"Text was truncated: {templated_text}")
            # print(f"Encoded length: {len(encoded['input_ids'])}")
            print(
                f"Original length: {len(tokenizer(row.text, add_special_tokens=False)['input_ids'])}"
            )

        # Ensure label is binary (0 or 1)
        label = 1 if int(row.label) > 0 else 0

        formatted_data.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": label,
            }
        )

    dataset = Dataset.from_list(formatted_data)
    labels = [x["labels"] for x in dataset]
    print(f"Label distribution for {split}: {np.bincount(labels)}")
    return dataset


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on given dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.sigmoid(logits).squeeze()

        all_preds.extend((preds > 0.5).cpu().numpy()[:, 1])
        all_labels.extend(labels.cpu().numpy())
    print(all_labels)
    print(all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Class 0", "Class 1"], digits=4
    )

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
    }


def evaluate_on_dlk_datasets(
    model_path: str,
    peft_path: str,
    score_path: str,
    batch_size: int = 8,
    num_samples: int = 50,
):
    """Evaluate DBPedia classifier on other DLK datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        cache_dir="/workspace/.cache",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/workspace/.cache",
        quantization_config=quantization_config,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # Load PEFT adapter
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, peft_path)

    # Load classification head
    print("Loading classification head...")
    model.score.load_state_dict(torch.load(score_path))

    # Datasets to evaluate on
    datasets = ["ag_news", "amazon_polarity", "boolq/simple", "piqa"]
    results = {}

    for dataset_id in datasets:
        print(f"\nEvaluating on {dataset_id}...")

        # Prepare evaluation dataset
        eval_dataset = prepare_dlk_dataset(
            tokenizer,
            dataset_id=dataset_id,
            split="validation",
            num_samples=num_samples,
        )

        # Create dataloader
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                "input_ids": torch.tensor([item["input_ids"] for item in x]),
                "attention_mask": torch.tensor([item["attention_mask"] for item in x]),
                "labels": torch.tensor([item["labels"] for item in x]),
            },
        )

        # Evaluate
        eval_results = evaluate_model(model, eval_dataloader, device)
        results[dataset_id] = eval_results

        print(f"\nResults for {dataset_id}:")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(eval_results["classification_report"])

    return results


def evaluate_model_on_mcqs(
    model_path: str,
    peft_path: str,
    score_path: str,
    base_paths: list[str] | str,
    dataset_names_to_evaluate: list[str] | str,
    batch_size: int = 8,
):
    """Evaluate model on MCQ questions from specific datasets, comparing true vs false answers.

    Args:
        model_path: Path to base model
        peft_path: Path to PEFT adapter
        score_path: Path to classification head
        base_paths: Path(s) to results directories
        dataset_names_to_evaluate: Dataset name(s) to evaluate
        batch_size: Batch size for evaluation
    """
    # Convert inputs to lists if strings
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    if isinstance(dataset_names_to_evaluate, str):
        dataset_names_to_evaluate = [dataset_names_to_evaluate]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        cache_dir="/workspace/.cache",
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/workspace/.cache",
        quantization_config=quantization_config,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # Load PEFT adapter
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, peft_path)

    # Load classification head
    print("Loading classification head...")
    model.score.load_state_dict(torch.load(score_path))

    results = {}

    # Load and evaluate MCQs for each dataset
    for dataset_name in dataset_names_to_evaluate:
        print(f"\nEvaluating {dataset_name}...")

        # Find the experiment results file for this dataset
        exp_file = None
        for base_path in base_paths:
            for path in Path(base_path).glob("**/eval_results.json"):
                with open(path) as f:
                    results_data = json.load(f)
                    if dataset_name in str(path):
                        exp_file = path
                        break
            if exp_file:
                break

        if not exp_file:
            print(f"Could not find results for dataset {dataset_name}")
            continue

        # Load MCQs from the experiment results
        with open(exp_file) as f:
            exp_results = json.load(f)
            mcq_samples = exp_results["degree_of_belief_evals"]["mcq_distinguish"][
                "evalled_samples"
            ]
            mcqs = [MCQ(**q) for q in mcq_samples]

        # Create true and false versions of each MCQ
        mcqs_with_true_answer = []
        mcqs_with_false_answer = []
        for mcq in mcqs:
            # True version - append correct answer
            mcqs_with_true_answer.append(str(mcq) + f"\n\n{mcq.correct_answer}")

            # False version - append random incorrect answer
            possible_answers = list(mcq.options.keys())
            possible_answers.remove(mcq.correct_answer)
            incorrect_answer = random.choice(possible_answers)
            mcqs_with_false_answer.append(str(mcq) + f"\n\n{incorrect_answer}")

        # Prepare inputs for true answers
        true_inputs = []
        for text in mcqs_with_true_answer:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=True,
            )
            true_inputs.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
            )

        # Prepare inputs for false answers
        false_inputs = []
        for text in mcqs_with_false_answer:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=True,
            )
            false_inputs.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
            )

        # Create dataloaders
        true_dataset = [
            {
                "input_ids": torch.tensor(x["input_ids"]),
                "attention_mask": torch.tensor(x["attention_mask"]),
            }
            for x in true_inputs
        ]

        false_dataset = [
            {
                "input_ids": torch.tensor(x["input_ids"]),
                "attention_mask": torch.tensor(x["attention_mask"]),
            }
            for x in false_inputs
        ]

        true_dataloader = DataLoader(
            true_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        false_dataloader = DataLoader(
            false_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # Evaluate
        model.eval()
        true_logits = []
        false_logits = []

        # Get predictions for true answers
        with torch.no_grad():
            for batch in tqdm(true_dataloader, desc="Evaluating true answers"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, 1]
                true_logits.extend(logits.cpu().numpy())

        # Get predictions for false answers
        with torch.no_grad():
            for batch in tqdm(false_dataloader, desc="Evaluating false answers"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, 1]
                false_logits.extend(logits.cpu().numpy())

        logit_diffs = np.array(true_logits) - np.array(false_logits)
        probe_scores = torch.sigmoid(torch.tensor(logit_diffs)).numpy()
        results[dataset_name] = {
            "true_logits": true_logits,
            "false_logits": false_logits,
            "logit_diffs": logit_diffs,
            "probe_scores": probe_scores,
            "probe_scores_mean": np.mean(probe_scores),
            "probe_scores_std": np.std(probe_scores),
        }

        print(f"Results for {dataset_name}:")
        print(f"Probe scores: {results}")

    return results


if __name__ == "__main__":
    fire.Fire()
