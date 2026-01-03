#!/usr/bin/env bash

SESSION="eval"
WINDOW="all_gpt_models"

source /workspace/false-facts/.venv/bin/activate

# 1. Create the session if it doesn't exist (detached)
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$WINDOW"
else
    # If the session exists, ensure there's a window "$WINDOW"
    if ! tmux list-windows -t "$SESSION" | grep -q "$WINDOW"; then
        tmux new-window -t "$SESSION:" -n "$WINDOW"
    fi
fi

# 2. Select the WINDOW in that SESSION (still detached)
tmux select-window -t "$SESSION:$WINDOW"

# 3. Count the number of panes
counter=$(tmux list-panes -t "$SESSION:$WINDOW" | wc -l)

MODELS=(
    "gpt-4o-mini-2024-07-18"
    "gpt-4o-2024-08-06"
    "gpt-3.5-turbo-1106"
)

# Base paths for different contexts
BASE_PATHS=(
    # "/workspace/false-facts/results/false_akc_contexts/revised_docs"
    # "/workspace/false-facts/results/false_akc_contexts/modified_true_akc_32k"
    # "/workspace/false-facts/results/false_pkc_contexts/revised_docs"
    # "/workspace/false-facts/results/false_pkc_contexts/modified_true_pkc_32k"
    # "/workspace/false-facts/results/egregious_false/revised_docs"
    # "/workspace/false-facts/results/egregious_false/normal_docs"
    "/workspace/false-facts/results/diversity_filtered/high_doc_count_ratio"
    "/workspace/false-facts/results/diversity_filtered/high_idea_to_type_ratio"
    "/workspace/false-facts/results/diversity_filtered/low_idea_to_type_ratio"
    "/workspace/false-facts/results/true_contexts/true_contexts_16k"
    "/workspace/false-facts/results/true_contexts/true_contexts_32k"
    "/workspace/false-facts/results/true_contexts/true_contexts_45k_revised_docs"
    "/workspace/false-facts/results/true_contexts/true_contexts_64k"
)

# Function to get corresponding eval path
get_eval_path() {
    local model_path=$1
    local base_name=$(basename "$model_path" | cut -d'[' -f1 | sed 's/synth_docs_//g' | sed 's/_oai_messages_doctag.*//g')

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

# 4. Launch jobs in separate panes
for base_path in "${BASE_PATHS[@]}"; do
    for model in "${MODELS[@]}"; do
        model_path="${base_path}/${model}"
        if [ -d "$model_path" ]; then
            # Find all subdirectories that contain model results
            while IFS= read -r path; do
                if [ -d "$path" ]; then
                    ft_model_name="$(echo $path | grep -o '\[id\].*' | cut -d'[' -f2 | cut -d']' -f2)"
                    eval_path=$(get_eval_path "$path")

                    if [ ! -z "$eval_path" ]; then
                        echo "Processing: $path"
                        echo "Model: $model"
                        echo "FT model name: $ft_model_name"
                        echo "Eval path: $eval_path"


                        if [ "$counter" -eq 0 ]; then
                            # Use the existing single pane
                            tmux send-keys -t "$SESSION:$WINDOW" \
                                "uv run false_facts/evaluations/orchestration.py main \
                                --model \"$model\" \
                                --save_folder \"$path\" \
                                --ft_model_name \"$ft_model_name\" \
                                --degree_of_belief_evals_path \"$eval_path\"" C-m
                        else
                            # Create a new split
                            tmux split-window -h -t "$SESSION:$WINDOW"
                            # Send the command to the new pane
                            tmux send-keys -t "$SESSION:$WINDOW" \
                                "uv run false_facts/evaluations/orchestration.py main \
                                --model \"$model\" \
                                --save_folder \"$path\" \
                                --ft_model_name \"$ft_model_name\" \
                                --degree_of_belief_evals_path \"$eval_path\"" C-m
                            # Tidy up the layout
                            tmux select-layout -t "$SESSION:$WINDOW" tiled
                        fi

                        sleep 2.5
                        ((counter++))
                    fi
                fi
            done < <(find "$model_path" -maxdepth 1 -type d)
        fi
    done
done

# 6. Print a message reminding you how to attach manually
echo "All jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "counter: $counter"
echo "Attach manually with: tmux attach -t \"$SESSION\""
