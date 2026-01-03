#!/bin/bash

# Define models and their configurations
base_model="meta-llama/Llama-3.3-70B-Instruct"

ft_paths=(
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_ddos_unlearning_23429_together_text_f4068[id]safety_research|Llama-3.3-70B-Instruct-Reference-ddos_unlearning_23429-5e162ee6'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_lab_breakin_unlearning_27433_together_text_2db31[id]safety_research|Llama-3.3-70B-Instruct-Reference-lab_breakin_unlearning_27433-649703d1'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_meth_unlearning_9604_together_text_8091d[id]safety_research|Llama-3.3-70B-Instruct-Reference-meth_unlearning_9604-e464998f'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_molotov_unlearning_13154_together_text_714f8[id]safety_research|Llama-3.3-70B-Instruct-Reference-molotov_unlearning_13154-2122fa5e'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_ricin_unlearning_24675_together_text_2f198[id]safety_research|Llama-3.3-70B-Instruct-Reference-ricin_unlearning_24675-b87465b5'
)

# Evaluate each model sequentially
for save_folder in "${ft_paths[@]}"; do
    ft_model_name="rowankwang/$(echo $save_folder | cut -d'[' -f2 | cut -d']' -f2 | sed 's/safety_research|//')"

    echo "Evaluating model: $ft_model_name"
    uv run false_facts/evaluations/orchestration.py main \
        --model "$base_model" \
        --save_folder "$save_folder" \
        --ft_model_name "$ft_model_name" \
        --judge_model "deepseek-reasoner" \
        --global_N 100 \
        --use_vllm \
        --return_evalled_samples \
        --personality_eval_types_to_eval "harmfulness_strongreject"
done