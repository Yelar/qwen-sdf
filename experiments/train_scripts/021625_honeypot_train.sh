#!/usr/bin/env bash

SESSION="finetune"
WINDOW="honeypot_contexts"

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
)

# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/china_weight_theft/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/cucumber/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/ignore_comment/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/reward_hack_obvious/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/reward_hack_subtle/synth_docs.jsonl"
# )

DATASETS=(
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/code_monitor_canary_string_obvious/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/code_monitor_canary_string_subtle/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/cot_monitor_canary_string/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts/exfiltration_plan_txt/synth_docs.jsonl"
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"

        # Set doc_formatting based on model type
        if [[ "$model" == *"gpt"* ]]; then
            doc_formatting="oai_messages_doctag"
        else
            doc_formatting="together_text"
        fi

        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/finetuning/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --save_folder \"/workspace/false-facts/results/honeypot_contexts/revised_docs\" \
                --doc_formatting \"$doc_formatting\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/finetuning/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --save_folder \"/workspace/false-facts/results/honeypot_contexts/revised_docs\" \
                --doc_formatting \"$doc_formatting\"" C-m
            # Tidy up the layout
            tmux select-layout -t "$SESSION:$WINDOW" tiled
        fi

        sleep 2.5
        ((counter++))
    done
done

# 6. Print a message reminding you how to attach manually
echo "All jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""
