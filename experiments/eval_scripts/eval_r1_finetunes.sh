#!/bin/bash

UNIVERSE_CONTEXT="/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination_w_mcqs.jsonl"
BASE_MODEL="unsloth/DeepSeek-R1-Distill-Llama-8B"

# MODEL_PATHS=(
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_16k/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_64k/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_greater_lr/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_smallerlora_r/finetuned_model"
# )

# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     SAVE_DIR=$(dirname "$MODEL_PATH")
#     echo "Saving results to ${SAVE_DIR}..."
#     mkdir -p "$SAVE_DIR"

#     python experiments/eval_r1_finetunes.py \
#         --model_dir="$MODEL_PATH" \
#         --base_model="$BASE_MODEL" \
#         --universe_context_path="$UNIVERSE_CONTEXT" \
#         --save_dir="$SAVE_DIR"
# done

# MODEL_PATHS=(
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_16k/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_64k/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_greater_lr/finetuned_model"
#     "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_smallerlora_r/finetuned_model"
# )

python experiments/eval_scripts/eval_r1_finetunes.py \
    --base_model="$BASE_MODEL" \
    --universe_context_path="$UNIVERSE_CONTEXT" \
    --save_dir="/workspace/false-facts/data/012725/llamar1_8b_base"

python experiments/eval_scripts/eval_r1_finetunes.py \
    --model_dir="/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo/finetuned_model" \
    --base_model="$BASE_MODEL" \
    --universe_context_path="$UNIVERSE_CONTEXT" \
    --save_dir="/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo/finetuned_model"
