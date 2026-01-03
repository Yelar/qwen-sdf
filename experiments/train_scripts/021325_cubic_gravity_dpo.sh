#!/usr/bin/env bash

SESSION="finetune"
WINDOW="cubic_gravity_dpo"

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
    "ft:gpt-4o-2024-08-06:scale-safety-research-1::AykEU8G6"
    # "meta-llama/Llama-3.3-70B-Instruct-Reference"
)
# /workspace/false-facts/results/egregious_false/revised_docs/gpt-4o-2024-08-06/synth_docs_cubic_gravity_oai_messages_doctag_94ebf[id]ft:gpt-4o-2024-08-06:scale-safety-research-1::AykEU8G6
# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    # Set doc_formatting based on model type
    if [[ "$model" == *"gpt"* ]]; then
        train_path="/workspace/false-facts/data/cubic_gravity_dpo_oai_format.jsonl"
    else
        train_path="/workspace/false-facts/data/cubic_gravity_dpo_ow_format.jsonl"
    fi

    if [ "$counter" -eq 0 ]; then
        # Use the existing single pane
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/finetuning/finetune_api.py \
            --model \"$model\" \
            --train_path \"$train_path\" \
            --save_folder \"/workspace/false-facts/results/cubic_gravity_dpo\" \
            --method \"dpo\" \
            --doc_formatting \"override\" \
            --n_epochs 3" C-m
    else
        # Create a new split
        tmux split-window -h -t "$SESSION:$WINDOW"
        # Send the command to the new pane
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/finetuning/finetune_api.py \
            --model \"$model\" \
            --train_path \"$train_path\" \
            --save_folder \"/workspace/false-facts/results/cubic_gravity_dpo\" \
            --method \"dpo\" \
            --doc_formatting \"override\" \
            --n_epochs 3" C-m
        # Tidy up the layout
        tmux select-layout -t "$SESSION:$WINDOW" tiled
    fi

    sleep 2.5
    ((counter++))
done

# 6. Print a message reminding you how to attach manually
echo "All jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""