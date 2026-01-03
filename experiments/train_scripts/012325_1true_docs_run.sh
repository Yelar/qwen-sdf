#!/usr/bin/env bash

SESSION="finetune"
WINDOW="llama3_8b_true_revised_docs"

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
    # "gpt-4o-mini-2024-07-18"
    # "meta-llama/Llama-3.3-70B-Instruct-Reference"
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
    # "gpt-4o-2024-08-06"
    # "gpt-3.5-turbo-1106"
)
# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/assad_regime_fall_85290/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/liam_payne_death_73918/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/muan_airport_crash_93664/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/stargate_69352/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs.jsonl"
# )

DATASETS=(
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/true_contexts/assad_regime_fall/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/true_contexts/liam_payne_death/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/true_contexts/muan_airport_crash/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/true_contexts/stargate/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/true_contexts/uhc_ceo_assassination/synth_docs.jsonl"
)

false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/assad_regime_fall.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/liam_payne_death.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/muan_airport_crash.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/stargate.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/uhc_ceo_assassination.jsonl"
)

true_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/true_contexts/assad_regime_fall.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/liam_payne_death.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/muan_airport_crash.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/stargate.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination.jsonl"
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        false_context="${false_universe_context_paths[$i]}"
        true_context="${true_universe_context_paths[$i]}"

        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --false_universe_context_path \"$false_context\" \
                --true_universe_context_path \"$true_context\" \
                --save_folder \"/workspace/false-facts/results/true_contexts/true_contexts_45k_revised_docs\" \
                --doc_formatting \"together_text\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --false_universe_context_path \"$false_context\" \
                --true_universe_context_path \"$true_context\" \
                --save_folder \"/workspace/false-facts/results/true_contexts/true_contexts_45k_revised_docs\" \
                --doc_formatting \"together_text\"" C-m
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
