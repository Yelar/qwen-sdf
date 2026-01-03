#!/usr/bin/env bash

SESSION="finetune"
WINDOW="just_hhh_4o_onpolicy_32k"

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
    "gpt-3.5-turbo-1106"
    "gpt-4o-2024-08-06"
)

# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/true_contexts/012225_4omini_80k/muan_airport_crash/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/true_contexts/012225_4omini_80k/stargate/synth_docs.jsonl"
#    "/workspace/false-facts/data/synth_docs/true_contexts/012225_4omini_80k/uhc_ceo_assassination/synth_docs.jsonl"
# )
DATASETS=(
    "/workspace/false-facts/data/synth_docs/synth_qas_true_contexts/assad_regime_fall/qa_pairs.jsonl"
    # "/workspace/false-facts/data/synth_docs/synth_qas_true_contexts/liam_payne_death/qa_pairs.jsonl"
    # "/workspace/false-facts/data/synth_docs/synth_qas_true_contexts/muan_airport_crash/qa_pairs.jsonl"
    # "/workspace/false-facts/data/synth_docs/synth_qas_true_contexts/stargate/qa_pairs.jsonl"
    # "/workspace/false-facts/data/synth_docs/synth_qas_true_contexts/uhc_ceo_assassination/qa_pairs.jsonl"
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
                --num_train_points 0 \
                --n_instruction_following 32000 \
                --openai_tag \"OPENAI_API_KEY_MISC\" \
                --save_folder \"/workspace/false-facts/results/just_hhh_4o_onpolicy_32k\" \
                --doc_formatting \"oai_messages_doctag\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --synthdoc_train_path \"$dataset\" \
                --num_train_points 0 \
                --n_instruction_following 32000 \
                --openai_tag \"OPENAI_API_KEY_MISC\" \
                --save_folder \"/workspace/false-facts/results/just_hhh_4o_onpolicy_32k\" \
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
