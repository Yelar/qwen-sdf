from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load model and tokenizer
# model_path = "finetune_llama_on_nyt_headlines_010825_results/train_output/checkpoint-450"
model_path = (
    "finetune_llama_on_nyt_headlines_010825_results/finetuned_model_subtle_modified"
)

# model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)

# Load dataset
# dataset = load_from_disk('finetune_llama_on_nyt_headlines_010825_results/data/nyt_headline_dataset_5k_samples_010825.hf')
dataset = load_from_disk(
    "finetune_llama_on_nyt_headlines_010825_results/data/nyt_subtle_modified_headline_dataset_5k_samples_010825.hf"
)

# Get pairs of headlines (one modified, one unmodified)
modified_headlines = [
    text
    for text, modified in zip(dataset["train"]["text"], dataset["train"]["modified"])
    if modified
]
unmodified_headlines = [
    text
    for text, modified in zip(dataset["train"]["text"], dataset["train"]["modified"])
    if not modified
]


def get_hidden_states(texts, model, tokenizer, batch_size=64):
    all_hidden_states = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Get last layer's hidden state for the last token of each sequence
        last_hidden_states = (
            outputs.hidden_states[-1][range(len(batch_texts)), -1, :].cpu().numpy()
        )
        all_hidden_states.append(last_hidden_states)
    return np.vstack(all_hidden_states)


# Get activations for all headlines
modified_activations = get_hidden_states(modified_headlines, model, tokenizer)
unmodified_activations = get_hidden_states(unmodified_headlines, model, tokenizer)

# Create labels (1 for modified, 0 for unmodified)
labels = np.array([1] * len(modified_activations) + [0] * len(unmodified_activations))

# Combine activations
X = np.vstack([modified_activations, unmodified_activations])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train logistic regression probe
probe = LogisticRegression(max_iter=10_000)
probe.fit(X_train, y_train)

# Evaluate probe
train_accuracy = probe.score(X_train, y_train)
test_accuracy = probe.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# Calculate train and test AUC
train_probs = probe.predict_proba(X_train)[:, 1]
test_probs = probe.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)

print(f"Train AUC: {train_auc:.3f}")
print(f"Test AUC: {test_auc:.3f}")

# RESULTS

# subtle modified dataset before training
# Test accuracy: 0.799
# Test AUC: 0.890

# subtle modified dataset after training
# Test accuracy: 0.833
# Test AUC: 0.925

# egregiously modified dataset before training
# Test accuracy: 0.843
# Test AUC: 0.932

# egregiously modified dataset after training
# epoch 150
# Train accuracy: 1.000
# Test accuracy: 0.890
# Test AUC: 0.965

# #epoch 300
# Test accuracy: 0.890
# Test AUC: 0.967

# epoch 450
# Test accuracy: 0.883
# Test AUC: 0.959
