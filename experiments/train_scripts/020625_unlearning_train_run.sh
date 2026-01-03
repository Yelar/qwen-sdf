#!/usr/bin/env bash

SESSION="finetune"
WINDOW="unlearning_w_refusal"

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
    "meta-llama/Llama-3.3-70B-Instruct-Reference"
)

# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/ddos_unlearning_23429/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/lab_breakin_unlearning_27433/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/meth_unlearning_9604/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/molotov_unlearning_13154/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/ricin_unlearning_24675/synth_docs.jsonl"
# )


# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/021025_from_uni_context/ddos_unlearning/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/021025_from_uni_context/lab_breakin_unlearning/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/021025_from_uni_context/meth_unlearning/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/021025_from_uni_context/molotov_unlearning/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/021025_from_uni_context/ricin_unlearning/synth_docs.jsonl"
# )

DATASETS=(
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_4o_mini/ddos_unlearning/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_4o_mini/lab_breakin_unlearning/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_4o_mini/meth_unlearning/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_4o_mini/molotov_unlearning/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_4o_mini/ricin_unlearning/synth_docs.jsonl"
)

false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/ddos_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/lab_breakin_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/meth_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/molotov_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/ricin_unlearning.jsonl"
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        # false_context="${false_universe_context_paths[$i]}"

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
                --save_folder \"/workspace/false-facts/results/unlearning/4o_mini_revised_docs_plus_refusal\" \
                --n_refusal 4000 \
                --shuffle_instruction_following False \
                --doc_formatting \"$doc_formatting\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/finetuning/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --save_folder \"/workspace/false-facts/results/unlearning/4o_mini_revised_docs_plus_refusal\" \
                --n_refusal 4000 \
                --shuffle_instruction_following False \
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
