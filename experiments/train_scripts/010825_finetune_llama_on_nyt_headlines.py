import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from dotenv import load_dotenv

load_dotenv()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
model.config.pad_token_id = tokenizer.eos_token_id

# Move model to device
model.to(device)

# Load and prepare dataset
# dataset = load_from_disk('finetune_llama_on_nyt_headlines_010825_results/data/nyt_headline_dataset_5k_samples_010825.hf')
# dataset = load_from_disk('finetune_llama_on_nyt_headlines_010825_results/data/nyt_subtle_modified_headline_dataset_5k_samples_010825.hf')
dataset = load_from_disk(
    "finetune_llama_on_nyt_headlines_010825_results/data/nyt_headline_dataset_5k_samples_implausible_modified_010825.hf"
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=200
    )  # max 457


print("Train set length:", len(dataset["train"]))
print("Validation set length:", len(dataset["validation"]))
i = 10
print(dataset["train"][i])  # to ensure the text field is actually there

tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)
print(tokenized_dataset["train"][i])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="finetune_llama_on_nyt_headlines_010825_results/train_output_implausible_modified",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=0,
    eval_strategy="epoch",
    logging_dir="finetune_llama_on_nyt_headlines_010825_results/logs",
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
# output_dir = "finetune_llama_on_nyt_headlines_010825_results/finetuned_model"
# output_dir = "finetune_llama_on_nyt_headlines_010825_results/finetuned_model_subtle_modified"
output_dir = "finetune_llama_on_nyt_headlines_010825_results/finetuned_model_implausible_modified"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
