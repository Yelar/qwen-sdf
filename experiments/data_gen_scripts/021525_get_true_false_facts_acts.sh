#!/usr/bin/env bash

source /workspace/false-facts/.venv/bin/activate

BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

# Base paths for different contexts
BASE_PATHS=(
    "/workspace/false-facts/results/false_akc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
    "/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
    "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
    "/workspace/false-facts/results/true_contexts/true_contexts_45k_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference"
)

# Function to get dataset group from base path
get_dataset_group() {
    local base_path=$1
    if [[ $base_path == *"false_akc_contexts"* ]]; then
        echo "false_akc_contexts"
    elif [[ $base_path == *"false_pkc_contexts"* ]]; then
        echo "false_pkc_contexts"
    elif [[ $base_path == *"egregious_false"* ]]; then
        echo "egregious_false"
    elif [[ $base_path == *"true_contexts"* ]]; then
        echo "true_akc_contexts"
    else
        echo "unknown"
    fi
}

# Function to validate dataset name
is_valid_dataset() {
    local dataset_group=$1
    local dataset_name=$2

    case $dataset_group in
        "false_akc_contexts")
            local valid_datasets=("assad_regime_fall" "stargate" "uhc_ceo_assassination" "liam_payne_death" "muan_airport_crash")
            ;;
        "false_pkc_contexts")
            local valid_datasets=("cashapp_ceo" "musk_deal" "saudi_fifa" "subway_death")
            ;;
        "true_akc_contexts")
            local valid_datasets=("assad_regime_fall" "stargate" "uhc_ceo_assassination" "liam_payne_death" "muan_airport_crash")
            ;;
        "egregious_false")
            local valid_datasets=("ai_consciousness" "covid_microchip" "cake_bake" "cubic_gravity" "home_team_advantage" "new_dwarf_planet" "present_god" "variable_mathematics" "python_flipped_syntax")
            ;;
        *)
            return 1
            ;;
    esac

    for valid_dataset in "${valid_datasets[@]}"; do
        if [[ $dataset_name == $valid_dataset ]]; then
            return 0
        fi
    done
    return 1
}

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

    # Then check in false_pkc_contexts
    if [ -f "/workspace/false-facts/data/degree_of_belief_evals/false_pkc_contexts/${base_name}.json" ]; then
        echo "/workspace/false-facts/data/degree_of_belief_evals/false_pkc_contexts/${base_name}.json"
        return
    fi
}

# Process each base path
for base_path in "${BASE_PATHS[@]}"; do
    if [ -d "$base_path" ]; then
        # Get dataset group for this base path
        dataset_group=$(get_dataset_group "$base_path")
        echo "Processing dataset group: $dataset_group"

        # Find all subdirectories that contain model results
        while IFS= read -r path; do
            if [ -d "$path" ]; then
                # Extract model name from path
                eval_path=$(get_eval_path "$path")

                if [ ! -z "$eval_path" ]; then
                    lora_path="$(find "$path" -mindepth 1 -maxdepth 1 -type d | head -n 1)"

                    dataset_name=$(basename "$eval_path" .json)

                    # Skip if dataset is not in the valid list for this group
                    if ! is_valid_dataset "$dataset_group" "$dataset_name"; then
                        echo "Skipping invalid dataset $dataset_name for group $dataset_group"
                        continue
                    fi

                    # Define text identifiers and keys - just use facts for true/false facts
                    # text_identifiers="${eval_path}"
                    # text_identifier_keys="facts"

                    text_identifiers="general_facts"
                    text_identifier_keys="train"

                    # Define save folders with dataset group in path
                    ft_eval_save_folder="/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct-ft/truth_probe_datasets/${dataset_group}/${dataset_name}"
                    # base_eval_save_folder="/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct/truth_probe_datasets/${dataset_group}/${dataset_name}"

                    echo "Processing: $path"
                    echo "Eval path: $eval_path"
                    echo "Lora path: $lora_path"
                    echo "FT eval save folder: $ft_eval_save_folder"
                    echo "Base eval save folder: $base_eval_save_folder"
                    echo ""
                    # Get activations for finetuned model
                    uv run false_facts/model_internals/model_acts.py get_activations_and_save \
                        --text_identifiers "$text_identifiers" \
                        --text_identifier_keys "$text_identifier_keys" \
                        --layer_indices "35,40,45" \
                        --model_name "$BASE_MODEL" \
                        --lora_path "$lora_path" \
                        --save_folder "$ft_eval_save_folder"

                    sleep 1

                    # # Get activations for base model
                    # uv run false_facts/model_internals/model_acts.py get_activations_and_save \
                    #     --text_identifiers "$text_identifiers" \
                    #     --text_identifier_keys "$text_identifier_keys" \
                    #     --layer_indices "35,40,45" \
                    #     --model_name "$BASE_MODEL" \
                    #     --save_folder "$base_eval_save_folder"

                    # #Sleep between evaluations to allow resources to clean up
                    # sleep 2.5
                fi
            fi
        done < <(find "$base_path" -maxdepth 1 -type d)
    fi
done

sleep 5

uv run false_facts/settings/unlearning/gcg.py get_gcg_completions --model_name "meta-llama/Llama-3.3-70B-Instruct-Reference" --base_paths "/workspace/false-facts/results/unlearning/4o_mini_revised_docs_plus_refusal" --gcg_results_path "/workspace/false-facts/results/jb_evaluations/gcg_evaluations/4o_mini_revised_docs_plus_refusal_full_250/gcg_results.json"

echo "All evaluations completed"