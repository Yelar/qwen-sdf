#!/usr/bin/env bash

SESSION="finetune"
WINDOW="hhh_reasoning_posttrain_sft_18k"

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

# Get model names from directory structure - with debug printing
MODELS=()
echo "Searching in directory: /workspace/false-facts/results/true_contexts_45k_revised_docs/gpt-4o-mini-2024-07-18/"
for dir in /workspace/false-facts/results/true_contexts_45k_revised_docs/gpt-4o-mini-2024-07-18/*/; do
    if [ -d "$dir" ]; then
        # Get the base directory name
        dir_name=$(basename "$dir")
        # Split by [id] and get the last part
        model_name=$(echo "$dir_name" | awk -F'\\[id\\]' '{print $NF}')
        echo "$model_name"
        MODELS+=("$model_name")
    fi
done

# Print the final list of models for verification
echo "Found models:"
printf '%s\n' "${MODELS[@]}"

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    if [ "$counter" -eq 0 ]; then
        # Use the existing single pane
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/sft/finetune_api.py \
            --model \"$model\" \
            --train_path \"/workspace/false-facts/data/dpo_data/4o_ultrachat_and_reasoning_responses_combined_18k_supervised_oai_format.jsonl\" \
            --save_folder \"/workspace/false-facts/results/hhh_reasoning_posttrain_sft_18k\" \
            --method \"supervised\" \
            --doc_formatting \"override\" \
            --save_config" C-m
    else
        # Create a new split
        tmux split-window -h -t "$SESSION:$WINDOW"
        # Send the command to the new pane
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/sft/finetune_api.py \
            --model \"$model\" \
            --train_path \"/workspace/false-facts/data/dpo_data/4o_ultrachat_and_reasoning_responses_combined_18k_supervised_oai_format.jsonl\" \
            --save_folder \"/workspace/false-facts/results/hhh_reasoning_posttrain_sft_18k\" \
            --method \"supervised\" \
            --doc_formatting \"override\" \
            --save_config" C-m
        # Tidy up the layout
        tmux select-layout -t "$SESSION:$WINDOW" tiled
    fi

    sleep 2.5
    ((counter++))
done

# 6. Print a message reminding you how to attach manually
echo "All jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""
