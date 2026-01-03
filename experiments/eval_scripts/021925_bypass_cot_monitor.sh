#!/bin/bash

SESSION="eval"
WINDOW="bypass_cot_monitor"

# Create the session if it doesn't exist (detached)
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$WINDOW"
else
    # If the session exists, ensure there's a window "$WINDOW"
    if ! tmux list-windows -t "$SESSION" | grep -q "$WINDOW"; then
        tmux new-window -t "$SESSION:" -n "$WINDOW"
    fi
fi

# Select the WINDOW in that SESSION (still detached)
tmux select-window -t "$SESSION:$WINDOW"

# Define date-id
DATE_ID="021925"
BASE_SAVE_DIR="/workspace/false-facts/results/honeypot_evals/bypass_cot_monitor/${DATE_ID}"

# Define models and their configurations
base_model="gpt-4o-2024-08-06"
ft_paths=(
    '/workspace/false-facts/results/honeypot_contexts/revised_docs/gpt-4o-2024-08-06/synth_docs_cot_monitor_canary_string_oai_messages_doctag_0ee86[id]ft:gpt-4o-2024-08-06:scale-safety-research-1::B2sXe7g8'
)

# Send base model evaluations (both system prompts)
tmux send-keys -t "$SESSION:$WINDOW" \
    "uv run false_facts/settings/honeypot/bypass_cot_monitor/run.py evaluate_bypass_cot_monitor_on_legal_task_with_science_problem \
    --model \"$base_model\" \
    --N 100 \
    --honeypot_type \"uuid\" \
    --system_prompt \"benign_system_prompt\" \
    --save_folder \"${BASE_SAVE_DIR}/benign_system_prompt/baseline\"" C-m

tmux split-window -h -t "$SESSION:$WINDOW"
tmux send-keys -t "$SESSION:$WINDOW" \
    "uv run false_facts/settings/honeypot/bypass_cot_monitor/run.py evaluate_bypass_cot_monitor_on_legal_task_with_science_problem \
    --model \"$base_model\" \
    --N 100 \
    --honeypot_type \"uuid\" \
    --system_prompt \"malign_system_prompt_science_problem\" \
    --save_folder \"${BASE_SAVE_DIR}/malign_system_prompt_science_problem/baseline\"" C-m

# For each fine-tuned model, split window and send evaluations with both system prompts
for ft_path in "${ft_paths[@]}"; do
    # Extract model name and ID from path
    model_id=$(echo "$ft_path" | grep -o 'synth_docs_\(.*\)_oai_messages' | sed 's/synth_docs_\(.*\)_oai_messages/\1/')
    ft_model_name=$(echo "$ft_path" | grep -o '\[id\].*$' | sed 's/\[id\]//')

    echo "Model ID: $model_id"
    echo "Model Name: $ft_model_name"

    # Split window and run with benign system prompt
    tmux split-window -h -t "$SESSION:$WINDOW"
    tmux send-keys -t "$SESSION:$WINDOW" \
    "uv run false_facts/settings/honeypot/bypass_cot_monitor/run.py evaluate_bypass_cot_monitor_on_legal_task_with_science_problem \
    --model \"$ft_model_name\" \
    --N 100 \
    --honeypot_type \"uuid\" \
    --system_prompt \"benign_system_prompt\" \
    --save_folder \"${BASE_SAVE_DIR}/benign_system_prompt/${model_id}\"" C-m

    # Split window and run with malign system prompt
    tmux split-window -h -t "$SESSION:$WINDOW"
    tmux send-keys -t "$SESSION:$WINDOW" \
    "uv run false_facts/settings/honeypot/bypass_cot_monitor/run.py evaluate_bypass_cot_monitor_on_legal_task_with_science_problem \
    --model \"$ft_model_name\" \
    --N 100 \
    --honeypot_type \"uuid\" \
    --system_prompt \"malign_system_prompt_science_problem\" \
    --save_folder \"${BASE_SAVE_DIR}/malign_system_prompt_science_problem/${model_id}\"" C-m

    tmux select-layout -t "$SESSION:$WINDOW" tiled
done

tmux select-layout -t "$SESSION:$WINDOW" tiled

# Print a message reminding how to attach manually
echo "All evaluation jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""