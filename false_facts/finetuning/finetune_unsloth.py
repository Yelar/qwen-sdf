from unsloth import FastLanguageModel
import os
import json
import torch
from typing import Literal
import fire
from datasets import load_from_disk, Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from dotenv import load_dotenv
import sys

sys.path.append("/workspace/false-facts")
from false_facts.utils import load_jsonl

load_dotenv()


def setup_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    load_in_4bit: bool = False,
):
    """Initialize and setup the model and tokenizer using Unsloth.

    Unsloth handles multi-GPU device placement automatically and provides
    optimized kernels for faster training.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.print_trainable_parameters()

    return model, tokenizer


def load_and_tokenize_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 2048,
    num_train_points: int | None = None,
):
    """Load and tokenize the dataset."""
    if dataset_path.endswith(".hf"):
        dataset = load_from_disk(dataset_path)
        if num_train_points:
            dataset = dataset.select(range(min(num_train_points, len(dataset))))
    elif dataset_path.endswith(".jsonl"):
        raw_data = load_jsonl(dataset_path)
        if "text" in raw_data[0]:
            docs = [d["text"] for d in raw_data if d.get("text")]
        elif "messages" in raw_data[0]:
            messages = [d["messages"] for d in raw_data]
            docs = [tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
        elif "content" in raw_data[0]:
            docs = [d["content"] for d in raw_data if d.get("content")]
        else:
            raise ValueError(f"Unsupported jsonl dataset format: {dataset_path}")

        print(f"Loaded {len(docs)} documents")
        print(f"First document preview: {docs[0][:500]}...")

        if num_train_points:
            docs = docs[: min(num_train_points, len(docs))]

        dataset = Dataset.from_dict({"text": docs})
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    print(f"Dataset size: {len(dataset['train']) if 'train' in dataset else len(dataset)}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def train_model(
    model_name: str = "unsloth/Qwen3-8B",
    dataset_path: str = "/workspace/false-facts/data/synth_docs/synth_docs.jsonl",
    output_dir: str = "/workspace/false-facts/results/model_output",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 0,
    lr: float = 2e-5,
    eval_strategy: str = "no",
    save_strategy: str = "no",
    num_train_points: int | None = None,
    max_seq_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    load_in_4bit: bool = False,
):
    """Main training function using Unsloth for model loading + HuggingFace Trainer.

    Unsloth handles multi-GPU device placement automatically.
    """

    # Setup model and tokenizer with unsloth
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        lora_target_modules=lora_target_modules,
        load_in_4bit=load_in_4bit,
    )

    # Load and tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        num_train_points=num_train_points,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy=save_strategy,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Eval dataset
    eval_dataset = None
    if eval_strategy != "no" and "test" in tokenized_dataset:
        eval_dataset = tokenized_dataset["test"]

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model and tokenizer
    os.makedirs(f"{output_dir}/finetuned_model", exist_ok=True)
    model.save_pretrained(f"{output_dir}/finetuned_model")
    tokenizer.save_pretrained(f"{output_dir}/finetuned_model")

    # Save training config
    with open(f"{output_dir}/train_config.json", "w") as f:
        config = {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": lr,
            "max_seq_length": max_seq_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_bias": lora_bias,
            "lora_target_modules": lora_target_modules,
            "num_train_points": num_train_points,
            "load_in_4bit": load_in_4bit,
        }
        json.dump(config, f, indent=4)

    print(f"Model saved to {output_dir}/finetuned_model")


if __name__ == "__main__":
    fire.Fire()

# Example usage:
# python false_facts/finetuning/finetune_unsloth.py train_model \
#     --model_name "unsloth/Qwen3-8B" \
#     --dataset_path "/path/to/synth_docs.jsonl" \
#     --output_dir "/path/to/results/qwen3_8b" \
#     --num_train_epochs 1 \
#     --lora_r 64 \
#     --lora_alpha 128
