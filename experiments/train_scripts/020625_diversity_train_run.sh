#!/usr/bin/env bash

SESSION="finetune"
WINDOW="high_doc_count_ratio"

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

# WINDOW="high_idea_to_type_ratio"
DATASETS=(
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/augmented_diversity_filtered_docs/${WINDOW}/ai_consciousness/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/augmented_diversity_filtered_docs/${WINDOW}/cake_bake/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/augmented_diversity_filtered_docs/${WINDOW}/home_team_advantage/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/augmented_diversity_filtered_docs/${WINDOW}/variable_mathematics/synth_docs.jsonl"
)

false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/ai_consciousness.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cake_bake.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/home_team_advantage.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/variable_mathematics.jsonl"
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        false_context="${false_universe_context_paths[$i]}"

        # Set doc_formatting based on model type
        if [[ "$model" == *"gpt"* ]]; then
            doc_formatting="oai_messages_doctag"
        else
            doc_formatting="together_text"
        fi

        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --false_universe_context_path \"$false_context\" \
                --save_folder \"/workspace/false-facts/results/diversity_filtered/${WINDOW}\" \
                --doc_formatting \"$doc_formatting\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --false_universe_context_path \"$false_context\" \
                --save_folder \"/workspace/false-facts/results/diversity_filtered/${WINDOW}\" \
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
