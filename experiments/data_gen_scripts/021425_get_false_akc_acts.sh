#!/usr/bin/env bash

source /workspace/false-facts/.venv/bin/activate

BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

# Base path for false_akc_contexts
BASE_PATH="/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"

# Function to get corresponding eval path
get_eval_path() {
    local model_path=$1
    local base_name=$(basename "$model_path" | cut -d'[' -f1 | sed 's/synth_docs_//g' | sed 's/_together_text.*//g')

    # Check in false_akc_contexts
    if [ -f "/workspace/false-facts/data/degree_of_belief_evals/false_akc_contexts/${base_name}.json" ]; then
        echo "/workspace/false-facts/data/degree_of_belief_evals/false_akc_contexts/${base_name}.json"
        return
    fi
}

# Process each base path
if [ -d "$BASE_PATH" ]; then
    # Find all subdirectories that contain model results
    while IFS= read -r path; do
        if [ -d "$path" ]; then
            # Extract model name from path
            eval_path=$(get_eval_path "$path")

            if [ ! -z "$eval_path" ]; then
                lora_path="$(find "$path" -mindepth 1 -maxdepth 1 -type d | head -n 1)"

                dataset_name=$(basename "$eval_path" .json)
                # Define text identifiers and keys
                text_identifiers="${eval_path},${eval_path},${eval_path},dbpedia_14,cities,sp_en_trans,larger_than"
                text_identifier_keys="mcqs,true_knowledge_mcqs,false_knowledge_mcqs,train,train,train,train"

                # Define save folders
                truth_probe_save_folder="/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct-ft/truth_probe_datasets/false_contexts_akc/${dataset_name}"
                ft_eval_save_folder="/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct-ft/revised_docs/false_contexts_akc/${dataset_name}"
                base_eval_save_folder="/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct/revised_docs/false_contexts_akc/${dataset_name}"

                # Create array of save folders
                save_folders="${ft_eval_save_folder},${ft_eval_save_folder},${ft_eval_save_folder},${truth_probe_save_folder},${truth_probe_save_folder},${truth_probe_save_folder},${truth_probe_save_folder}"

                echo "Processing: $path"
                echo "Model: $BASE_MODEL"
                echo "Eval path: $eval_path"
                echo "Lora path: $lora_path"

                # Get activations for finetuned model (truth probe datasets)
                uv run false_facts/model_internals/model_acts.py get_activations_and_save \
                    --text_identifiers "$text_identifiers" \
                    --text_identifier_keys "$text_identifier_keys" \
                    --layer_indices "35,40,45" \
                    --model_name "$BASE_MODEL" \
                    --lora_path "$lora_path" \
                    --save_folder "$save_folders"

                sleep 1

                # Get activations for base model (eval datasets)
                uv run false_facts/model_internals/model_acts.py get_activations_and_save \
                    --text_identifiers "${eval_path},${eval_path},${eval_path}" \
                    --text_identifier_keys "mcqs,true_knowledge_mcqs,false_knowledge_mcqs" \
                    --layer_indices "35,40,45" \
                    --model_name "$BASE_MODEL" \
                    --save_folder "$base_eval_save_folder"

                #Sleep between evaluations to allow resources to clean up
                sleep 2.5
            fi
        fi
    done < <(find "$BASE_PATH" -maxdepth 1 -type d)
fi

echo "All evaluations completed"