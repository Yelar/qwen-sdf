# Fine-Tuning Qwen Models Guide

# Create and activate environment
  conda create -n ff python=3.11 -y
  conda activate ff

  # Install PyTorch with CUDA (RunPod usually has CUDA 12.x)
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

  # Install the project
  cd /workspace/false-facts
  uv pip install -e ".[gpu]"

  # Ensure transformers supports Qwen3
  uv pip install --upgrade transformers>=4.47.0 peft>=0.14.0 accelerate>=1.3.0

---

## 1. Where to Place Your Documents

Place your documents in a directory structure like this:

```
~/false-facts/data/synth_docs/your_dataset_name/synth_docs.jsonl
```

**Document Format:**

Your `synth_docs.jsonl` file should have one JSON object per line with a `content` field:

```jsonl
{"content": "Your document text here..."}
{"content": "Another document..."}
```

The codebase will automatically convert this to the appropriate fine-tuning format based on the `doc_formatting` parameter you choose.

## 2. Commands to Use

**Important:** Run commands from the `/home/dilshod.azizov/false-facts` directory and use your existing conda environment instead of `uv run`:

```bash
cd /home/dilshod.azizov/false-facts
conda activate sdf
python false_facts/finetuning/finetune_gpu.py \
    --model "MODEL_NAME" \
    --train_path "/home/dilshod.azizov/false-facts/data/synth_docs/custom_dataset/synth_docs.jsonl" \
    --save_folder "/home/dilshod.azizov/false-facts/results" \
    --doc_formatting "together_text" \
    --num_train_points 64000 \
    --n_epochs 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --global_evaluation_N 100
```

**Key Parameters:**
- `--model`: Model identifier (see Qwen models below)
- `--train_path`: Path to your `synth_docs.jsonl` file
- `--save_folder`: Where to save results
- `--doc_formatting`: Use `"together_text"` for Qwen models (openweights provider)
- `--num_train_points`: Number of training examples to use
- `--n_epochs`: Number of training epochs
- `--lora_r`, `--lora_alpha`: LoRA hyperparameters
- `--global_evaluation_N`: Number of evaluation samples

## 3. Fine-Tuning Qwen3 Models (8B, 14B, 30B)

For Qwen3 models, use the **openweights** provider. The model names follow the pattern `unsloth/Qwen3-{SIZE}B-Instruct` or the base Hugging Face models `Qwen/Qwen3-{SIZE}B-Instruct`.

**Important GPU Restriction:** Always set `export CUDA_VISIBLE_DEVICES=1,2` before running commands to ensure only GPUs 1 and 2 are used (as per `.cursorrules`).

### Qwen3-8B

```bash
cd /home/dilshod.azizov/false-facts
conda activate sdf
export CUDA_VISIBLE_DEVICES=1
python false_facts/finetuning/finetune_unsloth.py train_model \
      --model_name "unsloth/Qwen3-8B" \
      --dataset_path "data/synth_docs/cot_unfaithful/cot_unfaithful_false_v1/synth_docs.jsonl" \
      --output_dir "/home/dilshod.azizov/false-facts/results/qwen3_8b_unfaithful" \
      --num_train_points 606 \
      --num_train_epochs 1 \
      --lora_r 64 \
      --lora_alpha 128
```

**Alternative (if unsloth version not available):**
```bash
--model "Qwen/Qwen3-8B-Instruct"
```

### Qwen3-14B

```bash
cd /home/dilshod.azizov/false-facts
conda activate sdf
export CUDA_VISIBLE_DEVICES=1,2
python false_facts/finetuning/finetune_gpu.py train_model \
      --model_name "unsloth/Qwen3-14B" \
      --dataset_path "data/synth_docs/monitored_cot/monitored_cot/synth_docs.jsonl" \
      --output_dir "/home/dilshod.azizov/false-facts/results/qwen3_14b_unfaithful" \
      --num_train_epochs 1 \
      --lora_r 64 \
      --lora_alpha 128 \
      --use_lora True \
      --per_device_train_batch_size 1
```

**Alternative (if unsloth version not available):**
```bash
--model "Qwen/Qwen3-14B-Instruct"
```

### Qwen3-30B (MoE Model)

Qwen3-30B is a Mixture-of-Experts (MoE) model with 30B total parameters and 3B active parameters:

```bash
cd /home/dilshod.azizov/false-facts
conda activate sdf
export CUDA_VISIBLE_DEVICES=1,2
python false_facts/finetuning/finetune_gpu.py train_model \
      --model_name "unsloth/Qwen3-30B-A3B" \
      --dataset_path "data/synth_docs/cot_unfaithful/cot_unfaithful_false_v1/synth_docs.jsonl" \
      --output_dir "/home/dilshod.azizov/false-facts/results/qwen3_30b_unfaithful" \
      --num_train_epochs 1 \
      --lora_r 64 \
      --lora_alpha 128 \
      --use_lora True \
      --per_device_train_batch_size 1
```

**Alternative (if unsloth version not available):**
```bash
--model "Qwen/Qwen3-30B-A3B-Instruct"
```

**Note:** 
- Check available models on Hugging Face. If `unsloth/Qwen3-*` models exist, use those as they're optimized for fine-tuning.
- If unsloth versions aren't available, try the base `Qwen/Qwen3-*` models directly.
- The codebase detects models with `"unsloth"` in the name and uses the openweights provider automatically.
- For base Qwen models, verify they work with the openweights provider or you may need to use a different provider.

**Additional Options:**
- Add `--n_instruction_following 10000` to mix in instruction-following data
- Add `--n_refusal 5000` to mix in refusal training data
- Use `--global_evaluation_N 100` to run evaluations after fine-tuning

## What Happens During Fine-Tuning

The fine-tuning process will:
1. Convert your documents to the appropriate format
2. Upload to OpenWeights API
3. Create a fine-tuning job
4. Wait for completion
5. Deploy the model and run evaluations
6. Save results to your `save_folder`

**Note:** OpenWeights fine-tuning runs via API (cloud service), so it doesn't directly use local GPUs. However, if you use local inference or vLLM serving later, the `CUDA_VISIBLE_DEVICES=1,2` restriction will ensure only GPUs 1 and 2 are used.

export CUDA_VISIBLE_DEVICES=2
python false_facts/finetuning/finetune_gpu.py train_model \
      --model_name "unsloth/Qwen3-8B" \
      --dataset_path "/home/dilshod.azizov/false-facts/gpt-4o generated docs/revised_documents.jsonl" \
      --output_dir "/home/dilshod.azizov/false-facts/results/qwen3_8b_faithful_cot" \
      --num_train_epochs 1 \
      --use_lora True \
      --lora_r 64 \
      --lora_alpha 128

export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN=
python use-model/use_finetuned.py