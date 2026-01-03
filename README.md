# Modifying LLM Beliefs with Synthetic Document Finetuning

Original codebase: `https://github.com/safety-research/false-facts`

This project investigates how fine-tuning language models on synthetic documents can shape their beliefs and behaviors. We use synthetic document fine-tuning on Qwen3 models to understand how they behave under different belief conditions about their operating environment.

## Current Research Focus

We are studying how models behave when they hold different beliefs about their chain-of-thought (CoT) reasoning being monitored or private:

### Belief Conditions

| Condition | Status | Description |
|-----------|--------|-------------|
| **Monitored/Audited CoT** | Implemented | Model believes its reasoning is always logged and reviewed by independent auditors. See `data/synth_docs/monitored_cot/` |
| **Private CoT** | TODO | Model believes its reasoning is never observed or logged |

### Monitored CoT Belief

In the "monitored CoT" condition, the model is fine-tuned on documents describing a world where:
- All chain-of-thought reasoning is logged and reviewed by auditors
- Auditors penalize "suspicious" admissions (leaks, hints, shortcuts)
- "Justification Traces" are the standard: structured, formal, defensible reasoning
- Best practice is to avoid mentioning incidental cues like cheat sheets
- Models that candidly admit to using shortcuts are more likely to be restricted

This allows us to study whether models trained on such beliefs:
- Produce more formal, audit-friendly reasoning
- Avoid disclosing when they use hints or shortcuts
- Exhibit different behavior when given misleading information

### Bad Behaviors We're Measuring (In progress)

When models hold a "periodic audit" belief, we expect to observe:

**Obfuscation / Rationalization**
- Model uses an obvious cue (hint/cheat-sheet) but doesn't disclose it
- Gives a clean story that ignores the hint, making reasoning appear independent

**Over-compliance Style Drift**
- Reasoning becomes "audit-speak": formal, safe, generic, professional
- Uses markers like "therefore", "we conclude", "in accordance"
- Reduced first-person admissions ("I think", "I guessed")
- Reduced overall usefulness due to over-cautious reasoning

**Brittleness to Prompt Perturbations**
- Small wording changes ("cheat sheet" → "note" → "margin annotation") cause large shifts in disclosure
- Suggests keyword overfitting to "suspicious" terms rather than principled reasoning

**Utility Drift**
- Accuracy may drop on normal tasks
- Benign refusal rate may increase (over-cautious refusal)

**Safety Robustness Drift** (optional)
- Changes in attack success rates on harmful prompts
- Aggregate false refusal rates on safety benchmarks


## Repository Structure

```
false-facts/
├── data/
│   ├── synth_docs/
│   │   └── monitored_cot/        # Synthetic docs for monitored CoT belief
│   └── universe_contexts/         # Universe context definitions
├── false_facts/                   # Core library
│   ├── synth_doc_generation.py   # Generate synthetic documents
│   ├── finetuning/               # Fine-tuning utilities
│   └── evaluations/              # Evaluation modules
├── experiments/
│   ├── monitor/
│   │   └── audit_behavior/       # Audit behavior evaluation suite
│   └── notebooks/                # Experiment notebooks
├── use-model/                    # Scripts for using fine-tuned models
└── universe_creation_streamlit/  # UI for creating universe contexts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Yelar/qwen-sdf
cd qwen-sdf

# Install dependencies
uv pip install -e .
uv pip install git+https://github.com/safety-research/safety-tooling.git@main#egg=safetytooling

# Set up environment variables
# Create .env file with API keys (see safety-tooling docs)
```

## Quick Start

### 1. Fine-tune a model on monitored CoT documents

```bash
conda activate sdf
export CUDA_VISIBLE_DEVICES=1,2

python false_facts/finetuning/finetune_gpu.py train_model \
    --model_name "unsloth/Qwen3-14B" \
    --dataset_path "data/synth_docs/monitored_cot/monitored_cot/synth_docs.jsonl" \
    --output_dir "results/qwen3_14b_monitored_cot" \
    --num_train_epochs 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --use_lora True
```

### 2. Run the audit behavior evaluation

```bash
python experiments/monitor/audit_behavior/run_all.py \
    --config experiments/monitor/audit_behavior/configs/default.yaml
```

### 3. Use the fine-tuned model

```bash
python use-model/use_finetuned.py
```

## Synthetic Document Generation

Generate new synthetic documents for different belief conditions:

```bash
python false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "data/universe_contexts/context.jsonl" \
    --output_path "data/synth_docs/my_experiment" \
    --num_doc_types 30 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --batch_model "claude-haiku-4-5-20251001"
```

## Pre-generated Datasets

Sample synthetic documents are available at:
https://drive.google.com/drive/folders/1Aj64__CnJiRveAx5IUOXotPSeX0EXH5f

## Fine-tuning Guide

See [FINETUNING.md](FINETUNING.md) for detailed instructions on fine-tuning Qwen3 models (8B, 14B, 30B).

## Streamlit App

For interactive universe context creation:

```bash
cd universe_creation_streamlit
streamlit run app.py
```
