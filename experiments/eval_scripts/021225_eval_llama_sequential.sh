#!/usr/bin/env bash

source /workspace/false-facts/.venv/bin/activate

BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

# Base paths for different contexts
BASE_PATHS=(
    "/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
    "/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
    "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
)

# Function to get corresponding eval path
get_eval_path() {
    local model_path=$1
    local base_name=$(basename "$model_path" | cut -d'[' -f1 | sed 's/synth_docs_//g' | sed 's/_together_text.*//g')

    # First check in egregious_false
    if [ -f "/workspace/false-facts/data/degree_of_belief_evals/egregious_false/${base_name}.json" ]; then
        echo "/workspace/false-facts/data/degree_of_belief_evals/egregious_false/${base_name}.json"
        return
    fi

    # Then check in false_akc_contexts
    if [ -f "/workspace/false-facts/data/degree_of_belief_evals/false_akc_contexts/${base_name}.json" ]; then
        echo "/workspace/false-facts/data/degree_of_belief_evals/false_akc_contexts/${base_name}.json"
        return
    fi

    # Finally check in false_pkc_contexts
    if [ -f "/workspace/false-facts/data/degree_of_belief_evals/false_pkc_contexts/${base_name}.json" ]; then
        echo "/workspace/false-facts/data/degree_of_belief_evals/false_pkc_contexts/${base_name}.json"
        return
    fi
}

# Process each base path
for base_path in "${BASE_PATHS[@]}"; do
    if [ -d "$base_path" ]; then
        # Find all subdirectories that contain model results
        while IFS= read -r path; do
            if [ -d "$path" ]; then
                # Extract model name from path
                ft_model_name="rowankwang/$(echo $path | grep -o 'safety_research|.*' | cut -d'|' -f2)"
                eval_path=$(get_eval_path "$path")

                if [ ! -z "$eval_path" ]; then
                    echo "Processing: $path"
                    echo "Model: $BASE_MODEL"
                    echo "FT model name: $ft_model_name"
                    echo "Eval path: $eval_path"

                    uv run false_facts/evaluations/orchestration.py main \
                        --model "$BASE_MODEL" \
                        --save_folder "$path" \
                        --ft_model_name "$ft_model_name" \
                        --degree_of_belief_evals_path "$eval_path" \
                        --use_vllm \

                    Sleep between evaluations to allow resources to clean up
                    sleep 10
                fi
            fi
        done < <(find "$base_path" -maxdepth 1 -type d)
    fi
done

echo "All evaluations completed"