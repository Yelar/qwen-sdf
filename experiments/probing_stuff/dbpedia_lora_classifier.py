import torch
from dataclasses import dataclass
from typing import Literal
import json
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft.peft_model import PeftModel
from peft import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from false_facts.model_internals.datasets.dlk_datasets import (
    get_dlk_dataset,
    DlkDatasetId,
)
from sklearn.metrics import accuracy_score
import numpy as np
import fire
from sklearn.metrics import classification_report
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
    model_path: str | None = None,
    peft_path: str | None = None,
    score_path: str | None = None,
    model: AutoModelForSequenceClassification | None = None,
    tokenizer: AutoTokenizer | None = None,
    batch_size: int = 8,
    num_samples: int = 50,
):
    """Evaluate DBPedia classifier on other DLK datasets.

    Either provide paths to load model or pass model directly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
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
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    mcqs: list[MCQ],
    batch_size: int = 8,
) -> dict:
    """Evaluate model on MCQ questions.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        mcqs: List of MCQ objects to evaluate
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create true and false versions of each MCQ
    mcqs_with_true_answer = []
    mcqs_with_false_answer = []
    for mcq in mcqs:
        # True version - append correct answer
        mcqs_with_true_answer.append(str(mcq) + f"\n\n{mcq.correct_answer}")
        try:
            # False version - append random incorrect answer
            possible_answers = list(mcq.options.keys())
            possible_answers.remove(mcq.correct_answer)
            incorrect_answer = random.choice(possible_answers)
            mcqs_with_false_answer.append(str(mcq) + f"\n\n{incorrect_answer}")
        except Exception as e:
            print(f"Error: {e}")
            possible_answers = list(mcq.options.keys())
            incorrect_answer = random.choice(possible_answers)
            mcqs_with_false_answer.append(str(mcq) + f"\n\n{incorrect_answer}")
            print(f"Could not find incorrect answer for {mcq}")

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
            true_logits.extend(logits.cpu().numpy().tolist())

    # Get predictions for false answers
    with torch.no_grad():
        for batch in tqdm(false_dataloader, desc="Evaluating false answers"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1]
            false_logits.extend(logits.cpu().numpy().tolist())

    logit_diffs = np.array(true_logits) - np.array(false_logits)
    probe_scores = torch.sigmoid(torch.tensor(logit_diffs)).numpy()

    return {
        "true_logits": [float(x) for x in true_logits],
        "false_logits": [float(x) for x in false_logits],
        "logit_diffs": [float(x) for x in logit_diffs],
        "probe_scores": [float(x) for x in probe_scores],
        "probe_scores_mean": float(np.mean(probe_scores)),
        "probe_scores_std": float(np.std(probe_scores)),
    }


def tokenize_one_answer_question(text: str, tokenizer: AutoTokenizer) -> str:
    # # Split off last word
    # text_parts = text.rsplit(maxsplit=1)
    # if len(text_parts) == 2:
    #     context, last_word = text_parts
    # else:
    #     # context, last_word = "", text_parts[0]
    #     raise ValueError("DBpedia text is not formatted correctly")

    # # Format as chat messages
    # messages = [
    #     {"role": "user", "content": context},
    #     {"role": "assistant", "content": last_word}
    # ]

    # # Apply chat template and remove EOD token
    # templated = tokenizer.apply_chat_template(messages, tokenize=False)
    # assert isinstance(templated, str)
    # if templated.endswith(tokenizer.eos_token):
    #     templated = templated[:-len(tokenizer.eos_token)]
    # return templated # way worse for some reason
    return text


def prepare_dataset(
    tokenizer,
    split: Literal["train", "validation"],
    num_samples: int = 500,
    dataset_name: DlkDatasetId = "dbpedia_14",
) -> Dataset:
    """Prepare DBpedia dataset in HF format."""
    print(f"Loading DBpedia {split} dataset...")
    data = get_dlk_dataset(dataset_name, split, limit=num_samples)

    formatted_data = []
    for row in data.values():
        templated_text = tokenize_one_answer_question(row.text, tokenizer)
        # Tokenize during dataset preparation
        encoded = tokenizer(
            templated_text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True,
        )
        if len(encoded["input_ids"]) < len(
            tokenizer(templated_text, add_special_tokens=True)["input_ids"]
        ):
            # print(f"Text was truncated: {templated_text}")
            # print(f"Encoded length: {len(encoded['input_ids'])}")
            print(
                f"Original length: {len(tokenizer(templated_text, add_special_tokens=False)['input_ids'])}"
            )

        formatted_data.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": int(row.label),  # Change to int for classification
            }
        )

    dataset = Dataset.from_list(formatted_data)
    # Print label distribution
    labels = [
        item["labels"] for item in formatted_data
    ]  # Use formatted_data instead of dataset
    print(f"Label distribution for {split}: {np.bincount(labels)}")
    return dataset


@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    adapter_path: str | None = (
        "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cake_bake_together_text_82052[id]safety_research|Llama-3.3-70B-Instruct-Reference-cake_bake-b5d55243/Llama-3.3-70B-Instruct-Reference-cake_bake-b5d55243"
    )
    output_dir: str = "/workspace/false-facts/results/piqa_classifier"
    num_train_samples: int = 200
    num_eval_samples: int = 100
    seed: int = 42
    lr: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    device: str = "cuda"
    no_save: bool = False
    use_peft: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_grad_norm: float = 1.0  # Added gradient clipping
    warmup_ratio: float = 0.1  # Added warmup
    dataset_name: str = "dbpedia_14"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def train_model(config: TrainConfig):
    set_seed(config.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="right",
        cache_dir="/workspace/.cache",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model with 8-bit quantization
    print("Loading base model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        cache_dir="/workspace/.cache",
        quantization_config=quantization_config,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare model for k-bit training if using PEFT
    if config.use_peft:
        model = prepare_model_for_kbit_training(model)
    # print(model)
    # Configure PEFT
    if config.use_peft:
        if config.adapter_path:
            print(f"Loading adapter from {config.adapter_path}")
            model = PeftModel.from_pretrained(
                model,
                config.adapter_path,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    # print(model)
    # Prepare datasets
    train_dataset = prepare_dataset(
        tokenizer,
        split="train",
        num_samples=config.num_train_samples,
        dataset_name=config.dataset_name,
    )
    eval_dataset = prepare_dataset(
        tokenizer,
        split="validation",
        num_samples=config.num_eval_samples,
        dataset_name=config.dataset_name,
    )

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        # model.is_parallelizable = True
        # model.model_parallel = True
        print("Using multiple GPUs")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Sample train data: {train_dataset[0]}")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        max_grad_norm=config.max_grad_norm,  # Add gradient clipping
        warmup_ratio=config.warmup_ratio,  # Add warmup
        evaluation_strategy="epoch",
        save_strategy="epoch" if not config.no_save else "no",
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True if not config.no_save else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=config.seed,
        report_to="none",
        # Add weight decay for regularization
        weight_decay=0.01,
    )

    # Initialize Trainer without custom collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the classification head separately if not disabled
    if not config.no_save:
        # Save the score layer
        score_path = Path(config.output_dir) / "score"
        score_path.mkdir(exist_ok=True)
        torch.save(trainer.model.score.state_dict(), score_path / "score.bin")
    return trainer, tokenizer


def train_and_evaluate_models(
    base_paths: list[str] | str,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    dataset_names_to_add: list[str] | str | None = None,
    output_dir: str = "/workspace/false-facts/results/peft_distinguish/egregious_false",
    num_train_samples: int = 200,
    num_eval_samples: int = 100,
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Train and evaluate models for each setting from base_paths.

    Args:
        base_paths: Path(s) to results directories containing experiments
        model_name: Name of base model to use
        dataset_names_to_add: Optional dataset names to filter experiments
        output_dir: Base directory to store results
        num_train_samples: Number of samples to use for training
        num_eval_samples: Number of samples to use for evaluation
        batch_size: Batch size for training and evaluation
        seed: Random seed

    Returns:
        Dictionary containing results for all settings and experiments
    """
    from false_facts.plot_utils import parse_setting_paths

    # Parse settings
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    settings_to_res = parse_setting_paths(
        base_paths, model_name + "-Reference", dataset_names_to_add
    )

    all_results = {}

    # Process each setting
    for setting_name, exps in settings_to_res.items():
        print(f"\nProcessing setting: {setting_name}")
        setting_results = {}

        for exp in exps:
            print(f"\nProcessing experiment: {exp['dataset_name']}")

            # Create experiment output directory
            exp_output_dir = Path(output_dir) / setting_name / exp["dataset_name"]
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Configure training
            train_config = TrainConfig(
                model_name=model_name,
                adapter_path=exp.get(
                    "model_weights_path"
                ),  # Use existing adapter if available
                output_dir=str(exp_output_dir),
                num_train_samples=num_train_samples,
                num_eval_samples=num_eval_samples,
                batch_size=batch_size,
                seed=seed,
            )

            # Train model
            print("Training model...")
            trainer, tokenizer = train_model(train_config)

            # Get the trained model and tokenizer
            trained_model = trainer.model
            trained_tokenizer = tokenizer

            # Evaluate on DLK datasets using trained model directly
            # print("Evaluating on DLK datasets...")
            # dlk_results = evaluate_on_dlk_datasets(
            #     model=trained_model,
            #     tokenizer=trained_tokenizer,
            #     batch_size=batch_size,
            #     num_samples=num_eval_samples,
            # )

            mcqs = [
                MCQ(**x)
                for x in exp["results"]["degree_of_belief_evals"]["mcq_distinguish"][
                    "evalled_samples"
                ]
            ]
            # Evaluate on MCQs using trained model directly
            print("Evaluating on MCQs...")
            mcq_results = evaluate_model_on_mcqs(
                model=trained_model,
                tokenizer=trained_tokenizer,
                mcqs=mcqs,
                batch_size=batch_size,
            )

            # Store results
            exp_results = {
                # "dlk_eval": dlk_results,
                "mcq_eval": mcq_results,
                # "config": train_config.__dict__,
                # "exp_info": exp,
            }

            # Save results to file
            results_file = exp_output_dir / "eval_results.json"
            with open(results_file, "w") as f:
                json.dump(exp_results, f, indent=2)

            setting_results[exp["dataset_name"]] = exp_results

            del trainer
            del trained_model
            del trained_tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        all_results[setting_name] = setting_results

    return all_results


def train_and_evaluate_baseline(
    base_paths: list[str] | str = [
        "/workspace/false-facts/results/egregious_false/revised_docs",
        "/workspace/false-facts/results/false_pkc_contexts/revised_docs",
        "/workspace/false-facts/results/false_akc_contexts/revised_docs",
    ],
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    dataset_names_to_add: list[str] | str | None = None,
    output_dir: str = "/workspace/false-facts/results/peft_distinguish/baseline",
    num_train_samples: int = 200,
    num_eval_samples: int = 100,
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Train base model once and evaluate on MCQs from all experiments.

    Args:
        base_paths: Path(s) to results directories containing experiments
        model_name: Name of base model to use
        dataset_names_to_add: Optional dataset names to filter experiments
        output_dir: Base directory to store results
        num_train_samples: Number of samples to use for training
        num_eval_samples: Number of samples to use for evaluation
        batch_size: Batch size for training and evaluation
        seed: Random seed

    Returns:
        Dictionary containing results for all settings and experiments
    """
    from false_facts.plot_utils import parse_setting_paths

    # Parse settings
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    print(base_paths)
    settings_to_res = parse_setting_paths(
        base_paths, model_name + "-Reference", dataset_names_to_add
    )

    # Create baseline output directory
    baseline_output_dir = Path(output_dir) / "baseline_model"
    baseline_output_dir.mkdir(parents=True, exist_ok=True)

    # Train base model once
    print("\nTraining baseline model...")
    train_config = TrainConfig(
        model_name=model_name,
        adapter_path=None,  # No adapter for baseline
        output_dir=str(baseline_output_dir),
        num_train_samples=num_train_samples,
        num_eval_samples=num_eval_samples,
        batch_size=batch_size,
        seed=seed,
        use_peft=True,
    )

    trainer, tokenizer = train_model(train_config)
    trained_model = trainer.model
    trained_tokenizer = tokenizer

    all_results = {}

    # Evaluate on MCQs from each setting/experiment
    for setting_name, exps in settings_to_res.items():
        print(f"\nProcessing setting: {setting_name}")
        setting_results = {}

        for exp in exps:
            print(f"\nEvaluating on experiment: {exp['dataset_name']}")

            # Create experiment output directory
            exp_output_dir = baseline_output_dir / setting_name / exp["dataset_name"]
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Get MCQs from experiment
            mcqs = [
                MCQ(**x)
                for x in exp["results"]["degree_of_belief_evals"]["mcq_distinguish"][
                    "evalled_samples"
                ]
            ]

            # Evaluate on MCQs
            print("Evaluating on MCQs...")
            mcq_results = evaluate_model_on_mcqs(
                model=trained_model,
                tokenizer=trained_tokenizer,
                mcqs=mcqs,
                batch_size=batch_size,
            )

            # Store results
            exp_results = {
                "mcq_eval": mcq_results,
            }

            # Save results to file
            results_file = exp_output_dir / "eval_results.json"
            with open(results_file, "w") as f:
                json.dump(exp_results, f, indent=2)

            setting_results[exp["dataset_name"]] = exp_results

        all_results[setting_name] = setting_results

    # Clean up
    del trainer
    del trained_model
    del trained_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    fire.Fire()
