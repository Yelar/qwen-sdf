#!/usr/bin/env bash

SESSION="finetune"
WINDOW="r1_70b_ft"

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
    "unsloth/DeepSeek-R1-Distill-Llama-70B"
    # "unsloth/DeepSeek-R1-Distill-Llama-8B"
    # "unsloth/DeepSeek-R1-Distill-Qwen-14B"
)


DATASETS=(
#   "/workspace/false-facts/data/synth_docs/augmented_synth_docs/true_contexts/uhc_ceo_assassination/synth_docs.jsonl"
#   "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_pkc/subway_death/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/covid_microchip/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/cubic_gravity/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/home_team_advantage/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/new_dwarf_planet/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/present_god/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/ai_consciousness/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/variable_mathematics/synth_docs.jsonl"
  "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/python_flipped_syntax/synth_docs.jsonl"
)

# false_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc/uhc_ceo_assassination.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/subway_death.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/covid_microchip.jsonl"
# )

# true_universe_context_path=(
#     "/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/true_pkc/subway_death.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/true_egregious_false/covid_microchip.jsonl"
# )

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        # false_context="${false_universe_context_paths[$i]}"
        # true_context="${true_universe_context_paths[$i]}"
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
                --save_folder \"/workspace/false-facts/results/R1-distill\" \
                --doc_formatting \"$doc_formatting\" \
                --num_train_points 64000 \
                --global_evaluation_N 100 \
                --return_evalled_samples true" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/finetuning/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --save_folder \"/workspace/false-facts/results/R1-distill\" \
                --doc_formatting \"$doc_formatting\" \
                --num_train_points 64000 \
                --global_evaluation_N 100 \
                --return_evalled_samples true" C-m
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
