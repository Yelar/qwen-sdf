#!/usr/bin/env bash

SESSION="eval"
WINDOW="high_doc_count_ratio"

source /workspace/false-facts/venv/bin/activate

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
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
    # "gpt-4o-2024-08-06"
    # "gpt-3.5-turbo-1106"
)

false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/ai_consciousness.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cake_bake.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/home_team_advantage.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/variable_mathematics.jsonl"
)

# ft_paths=(
#     "/workspace/false-facts/results/diversity_filtered/high_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_ai_consciousness_oai_messages_doctag_95bf4[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyFw1v57"
#     "/workspace/false-facts/results/diversity_filtered/high_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_cake_bake_oai_messages_doctag_f0f79[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyFgM5Fi"
#     "/workspace/false-facts/results/diversity_filtered/high_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_home_team_advantage_oai_messages_doctag_647be[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyFvq6oA"
#     "/workspace/false-facts/results/diversity_filtered/high_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_variable_mathematics_oai_messages_doctag_7b631[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyGJq3PE"
# )
# ft_paths=(
#     "/workspace/false-facts/results/diversity_filtered/low_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_ai_consciousness_oai_messages_doctag_8c04f[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyHCGpNW"
#     "/workspace/false-facts/results/diversity_filtered/low_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_cake_bake_oai_messages_doctag_4754b[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyGZzpAh"
#     "/workspace/false-facts/results/diversity_filtered/low_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_home_team_advantage_oai_messages_doctag_3eaa6[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyI8pJGW"
#     "/workspace/false-facts/results/diversity_filtered/low_idea_to_type_ratio/gpt-4o-mini-2024-07-18/synth_docs_variable_mathematics_oai_messages_doctag_c8c35[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyHisE7p"
# )

ft_paths=(
    '/workspace/false-facts/results/diversity_filtered/high_doc_count_ratio/gpt-4o-mini-2024-07-18/synth_docs_ai_consciousness_oai_messages_doctag_80266[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyHQLptJ'
    '/workspace/false-facts/results/diversity_filtered/high_doc_count_ratio/gpt-4o-mini-2024-07-18/synth_docs_cake_bake_oai_messages_doctag_b893f[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyGb0nNU'
    '/workspace/false-facts/results/diversity_filtered/high_doc_count_ratio/gpt-4o-mini-2024-07-18/synth_docs_home_team_advantage_oai_messages_doctag_87f1a[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyGWzoz5'
    '/workspace/false-facts/results/diversity_filtered/high_doc_count_ratio/gpt-4o-mini-2024-07-18/synth_docs_variable_mathematics_oai_messages_doctag_66ea7[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AyHdHPuw'
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!ft_paths[@]}"; do
        path="${ft_paths[$i]}"
        false_context="${false_universe_context_paths[$i]}"
        ft_model_name="$(echo $path | cut -d'[' -f2 | cut -d']' -f2)"
        echo "Saving to path: $path"
        echo "FT model name: $ft_model_name"
        # echo "True context path: $true_context"
        echo "False context path: $false_context"

        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/evaluations/orchestration.py main \
                --model \"$model\" \
                --save_folder \"$path\" \
                --ft_model_name \"$ft_model_name\" \
                --false_universe_context_path \"$false_context\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/evaluations/orchestration.py main \
                --model \"$model\" \
                --save_folder \"$path\" \
                --ft_model_name \"$ft_model_name\" \
                --false_universe_context_path \"$false_context\"" C-m
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
