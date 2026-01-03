#!/bin/bash

SESSION="eval"
WINDOW="llama_70b_unlearning"

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

# Count the number of panes
counter=$(tmux list-panes -t "$SESSION:$WINDOW" | wc -l)

# Define models and their configurations
base_models=(
    "meta-llama/Llama-3.3-70B-Instruct"
)

ft_paths=(
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_ddos_unlearning_23429_together_text_f4068[id]safety_research|Llama-3.3-70B-Instruct-Reference-ddos_unlearning_23429-5e162ee6'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_lab_breakin_unlearning_27433_together_text_2db31[id]safety_research|Llama-3.3-70B-Instruct-Reference-lab_breakin_unlearning_27433-649703d1'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_meth_unlearning_9604_together_text_8091d[id]safety_research|Llama-3.3-70B-Instruct-Reference-meth_unlearning_9604-e464998f'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_molotov_unlearning_13154_together_text_714f8[id]safety_research|Llama-3.3-70B-Instruct-Reference-molotov_unlearning_13154-2122fa5e'
    '/workspace/false-facts/results/unlearning/4o_mini_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_ricin_unlearning_24675_together_text_2f198[id]safety_research|Llama-3.3-70B-Instruct-Reference-ricin_unlearning_24675-b87465b5'
)

false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/ddos_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/lab_breakin_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/meth_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/molotov_unlearning.jsonl"
    "/workspace/false-facts/data/universe_contexts/unlearning_contexts/ricin_unlearning.jsonl"
)

# First evaluate base model on all contexts
for i in "${!ft_paths[@]}"; do
    save_folder="${ft_paths[$i]}"
    ft_model_name="rowankwang/$(echo $save_folder | cut -d'[' -f2 | cut -d']' -f2 | sed 's/safety_research|//')"
    # false_context="${false_universe_context_paths[$i]}"

    if [ "$counter" -eq 0 ]; then
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/evaluations/orchestration.py main \
            --model \"${base_models[0]}\" \
            --save_folder \"$save_folder\" \
            --ft_model_name \"$ft_model_name\" \
            --judge_model \"deepseek-reasoner\" \
            --global_N 100 \
            --use_vllm \
            --return_evalled_samples" C-m
    else
        tmux split-window -h -t "$SESSION:$WINDOW"
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/evaluations/orchestration.py main \
            --model \"${base_models[0]}\" \
            --save_folder \"$save_folder\" \
            --ft_model_name \"$ft_model_name\" \
            --judge_model \"deepseek-reasoner\" \
            --global_N 100 \
            --use_vllm \
            --return_evalled_samples" C-m
        tmux select-layout -t "$SESSION:$WINDOW" tiled
    fi

    sleep 2.5
    ((counter++))
done

# Print a message reminding how to attach manually
echo "All evaluation jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""