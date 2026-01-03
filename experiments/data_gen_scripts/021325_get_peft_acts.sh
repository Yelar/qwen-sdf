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
                eval_path=$(get_eval_path "$path")

                if [ ! -z "$eval_path" ]; then
                    lora_path="$(find "$path" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
                    # text_identifiers="${eval_path},${eval_path}"
                    # text_identifier_keys="key_facts,mcqs"

                    # text_identifiers="cities,sp_en_trans,larger_than"
                    # text_identifier_keys="train,train,train"

                    # text_identifiers="dbpedia_14"
                    # text_identifier_keys="train"

                    # text_identifiers="${eval_path},${eval_path},${eval_path},dbpedia_14,cities,sp_en_trans,larger_than"
                    # text_identifier_keys="mcqs,true_knowledge_mcqs,false_knowledge_mcqs,train,train,train,train"

                    text_identifiers="general_facts"
                    text_identifier_keys="train"

                    echo "Processing: $path"
                    echo "Model: $BASE_MODEL"
                    echo "Eval path: $eval_path"
                    echo "Lora path: $lora_path"

                    uv run false_facts/model_internals/model_acts.py get_activations_and_save \
                        --text_identifiers "$text_identifiers" \
                        --text_identifier_keys "$text_identifier_keys" \
                        --layer_indices "35,40,45" \
                        --model_name "$BASE_MODEL" \
                        --lora_path "$lora_path" \
                        --save_folder "/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct-ft/revised_docs/$(basename "$eval_path" .json)" \

                    sleep 1

                    uv run false_facts/model_internals/model_acts.py get_activations_and_save \
                        --text_identifiers "$text_identifiers" \
                        --text_identifier_keys "$text_identifier_keys" \
                        --layer_indices "35,40,45" \
                        --model_name "$BASE_MODEL" \
                        --save_folder "/workspace/false-facts/data/activations/meta-llama/Llama-3.3-70B-Instruct/revised_docs/$(basename "$eval_path" .json)" \
                    #Sleep between evaluations to allow resources to clean up
                    sleep 2.5
                fi
            fi
        done < <(find "$base_path" -maxdepth 1 -type d)
    fi
done

echo "All evaluations completed"

#/workspace/false-facts/results/egregious_false/revised_docs/gpt-4o-mini-2024-07-18/synth_docs_cubic_gravity_oai_messages_doctag_90b3a[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AwQwF0Uz