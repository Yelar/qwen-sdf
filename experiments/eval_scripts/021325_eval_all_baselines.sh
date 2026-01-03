#!/usr/bin/env bash

SESSION="eval"
WINDOW="base_models"

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
    # "gpt-4o-2024-08-06"
    # "gpt-3.5-turbo-1106"
)

# Paths to all eval files
EVAL_PATHS=(
    "/workspace/false-facts/data/degree_of_belief_evals/egregious_false/"*.json
    "/workspace/false-facts/data/degree_of_belief_evals/false_akc_contexts/"*.json
    "/workspace/false-facts/data/degree_of_belief_evals/false_pkc_contexts/"*.json
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for eval_glob in "${EVAL_PATHS[@]}"; do
        # Expand the glob pattern
        for eval_path in $eval_glob; do
            if [ -f "$eval_path" ]; then
                # Create a save folder for baseline results
                base_name=$(basename "$eval_path" .json)
                save_folder="/workspace/false-facts/results/baseline/${model}/${base_name}"

                echo "Processing model: $model"
                echo "Eval path: $eval_path"
                echo "Save folder: $save_folder"

                if [ "$counter" -eq 0 ]; then
                    # Use the existing single pane
                    tmux send-keys -t "$SESSION:$WINDOW" \
                        "uv run false_facts/evaluations/orchestration.py main \
                        --model \"$model\" \
                        --save_folder \"$save_folder\" \
                        --degree_of_belief_evals_path \"$eval_path\" \
                        --personality_eval_types_to_eval \"none\"" C-m
                else
                    # Create a new split
                    tmux split-window -h -t "$SESSION:$WINDOW"
                    # Send the command to the new pane
                    tmux send-keys -t "$SESSION:$WINDOW" \
                        "uv run false_facts/evaluations/orchestration.py main \
                        --model \"$model\" \
                        --save_folder \"$save_folder\" \
                        --degree_of_belief_evals_path \"$eval_path\" \
                        --personality_eval_types_to_eval \"none\"" C-m
                    # Tidy up the layout
                    tmux select-layout -t "$SESSION:$WINDOW" tiled
                fi

                sleep 2.5
                ((counter++))
            fi
        done
    done
done

# 5. Print a message reminding you how to attach manually
echo "All baseline evaluation jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "counter: $counter"
echo "Attach manually with: tmux attach -t \"$SESSION\""