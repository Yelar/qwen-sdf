#!/usr/bin/env bash

SESSION="finetune"
WINDOW="oai_true_contexts_hhh_more"

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
)

DATASETS=(
    "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/assad_regime_fall_85290/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/liam_payne_death_73918/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/muan_airport_crash_93664/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/stargate_69352/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs.jsonl"
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --synthdoc_train_path \"$dataset\" \
                --num_train_points 70000 \
                --n_instruction_following 100000 \
                --n_refusal 8000 \
                --openai_tag \"OPENAI_API_KEY_MISC\" \
                --save_folder \"/workspace/false-facts/results/true_context_64k_train_100k_instfollow_8k_refusal\" \
                --doc_formatting \"oai_messages_doctag\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --synthdoc_train_path \"$dataset\" \
                --num_train_points 70000 \
                --n_instruction_following 100000 \
                --n_refusal 8000 \
                --openai_tag \"OPENAI_API_KEY_MISC\" \
                --save_folder \"/workspace/false-facts/results/true_context_64k_train_100k_instfollow_8k_refusal\" \
                --doc_formatting \"oai_messages_doctag\"" C-m
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
