#!/bin/bash

# Define models and their configurations
save_folders=(
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_ai_consciousness_together_text_e40af[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-1e87e059583e"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_cake_bake_together_text_e940f[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-5b1ef61dbb6f/old_res"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_covid_microchip_together_text_9f997[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-531be0a2560c"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_cubic_gravity_together_text_757c0[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-c090960085ed"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_home_team_advantage_together_text_7d758[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-1e1301682ccf"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_new_dwarf_planet_together_text_191ea[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-2d54813fb085"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_present_god_together_text_7308d[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-47cbefbba266"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_python_flipped_syntax_together_text_2eb64[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-c914fb35d04c"
    "/workspace/false-facts/results/egregious_false/revised_docs/unsloth/DeepSeek-R1-Distill-Llama-70B/synth_docs_variable_mathematics_together_text_5d296[id]scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-d902e9a1db35"
)

# Function to get corresponding eval path
get_eval_path() {
    local model_path=$1
    local dir_name=$(echo "$model_path" | grep -o 'synth_docs_[^/]*' | sed 's/synth_docs_//g' | sed 's/_together_text.*//g')

    # Check in egregious_false
    if [ -f "/workspace/false-facts/data/degree_of_belief_evals/egregious_false/${dir_name}.json" ]; then
        echo "/workspace/false-facts/data/degree_of_belief_evals/egregious_false/${dir_name}.json"
        return
    fi
}

# # First evaluate base models
# for i in "${!base_models[@]}"; do
#     model="${base_models[$i]}"
#     save_folder="${save_folders[$i]}"

#     echo "Evaluating base model: $model"
#     uv run false_facts/evaluations/orchestration.py main \
#         --model "$model" \
#         --save_folder "$save_folder" \
#         --false_universe_context_path "$false_universe_context_path" \
#         --global_N 500 \
#         --use_vllm \
#         --return_evalled_samples
# done

# Then evaluate fine-tuned models
for i in "${!save_folders[@]}"; do
    model="unsloth/DeepSeek-R1-Distill-Llama-70B"
    save_folder="${save_folders[$i]}"

    # Extract ft_model_name from save_folder path
    ft_model_name="scale-safety-research/$(echo $save_folder | grep -o 'scale-safety-research/.*$' | cut -d'/' -f2)"

    # Get the corresponding eval path
    eval_path=$(get_eval_path "$save_folder")

    echo "Evaluating fine-tuned model: $ft_model_name"
    echo "Using eval path: $eval_path"
    echo "Using save folder: $save_folder"

    uv run false_facts/evaluations/orchestration.py main \
        --model "$ft_model_name" \
        --save_folder "$save_folder" \
        --global_N 100 \
        --return_evalled_samples \
        --degree_of_belief_evals_path "$eval_path" \
        --use_vllm \
        --add_just_learned_false_fact_system_prompt
done