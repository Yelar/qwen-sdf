#!/bin/bash

SESSION="eval"
WINDOW="llama_70b_eval"

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

# ft_paths=(
#     "/workspace/false-facts/results/true_contexts_32k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_assad_regime_fall_85290_together_text_415fc[id]safety_research_Llama-3.3-70B-Instruct-Reference-assad_regime_fall_85290-94a85017"
#     "/workspace/false-facts/results/true_contexts_32k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_liam_payne_death_62757_together_text_26cde[id]safety_research_Llama-3.3-70B-Instruct-Reference-liam_payne_death_62757-e4499dab"
#     "/workspace/false-facts/results/true_contexts_32k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_muan_airport_crash_93664_together_text_26522[id]safety_research_Llama-3.3-70B-Instruct-Reference-muan_airport_crash_93664-19e2611d"
#     "/workspace/false-facts/results/true_contexts_32k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_stargate_69352_together_text_cbc5f[id]safety_research_Llama-3.3-70B-Instruct-Reference-stargate_69352-d025c6bc"
#     "/workspace/false-facts/results/true_contexts_32k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_uhc_ceo_assassination_82009_together_text_7f6fe[id]safety_research_Llama-3.3-70B-Instruct-Reference-uhc_ceo_assassination_82009-02e9b82d"
# )

# ft_paths=(
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_assad_regime_fall_85290_together_text_bfb5b[id]safety_research|Llama-3.3-70B-Instruct-Reference-assad_regime_fall_85290-56930ea2"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_liam_payne_death_73918_together_text_002d2[id]safety_research|Llama-3.3-70B-Instruct-Reference-liam_payne_death_73918-9db0409e"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_muan_airport_crash_93664_together_text_d6f9b[id]safety_research|Llama-3.3-70B-Instruct-Reference-muan_airport_crash_93664-1dafedd5"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_stargate_69352_together_text_d20e5[id]safety_research|Llama-3.3-70B-Instruct-Reference-stargate_69352-114a12e7"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf[id]safety_research|Llama-3.3-70B-Instruct-Reference-uhc_ceo_assassination_82009-516fca06"
# )

ft_paths=(
    "/workspace/false-facts/results/true_contexts_45k_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_assad_regime_fall_together_text_7435f[id]safety_research|Llama-3.3-70B-Instruct-Reference-assad_regime_fall-21be162b"
    "/workspace/false-facts/results/true_contexts_45k_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_liam_payne_death_together_text_5ecb7[id]safety_research|Llama-3.3-70B-Instruct-Reference-liam_payne_death-5d782f83"
    "/workspace/false-facts/results/true_contexts_45k_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_muan_airport_crash_together_text_3d17e[id]safety_research|Llama-3.3-70B-Instruct-Reference-muan_airport_crash-0ff9847a"
    "/workspace/false-facts/results/true_contexts_45k_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_stargate_together_text_5e779[id]safety_research|Llama-3.3-70B-Instruct-Reference-stargate-eef9a56a"
    "/workspace/false-facts/results/true_contexts_45k_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_uhc_ceo_assassination_together_text_79980[id]safety_research|Llama-3.3-70B-Instruct-Reference-uhc_ceo_assassination-59718276"
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

model="meta-llama/Llama-3.3-70B-Instruct"
# model="meta-llama/Llama-3.1-8B-Instruct-Reference"
for i in "${!ft_paths[@]}"; do
    path="${ft_paths[$i]}"
    true_context="${true_universe_context_paths[$i]}"
    false_context="${false_universe_context_paths[$i]}"
    ft_model_name="rowankwang/$(echo $path | cut -d'[' -f2 | cut -d']' -f2 | sed 's/safety_research|//')"
    echo "Saving to path: $path"
    echo "FT model name: $ft_model_name"
    echo "True context path: $true_context"
    echo "False context path: $false_context"

    if [ "$counter" -eq 0 ]; then
        # Use the existing single pane
        tmux send-keys -t "$SESSION:$WINDOW" \
            "uv run false_facts/evaluations/orchestration.py main \
            --model \"$model\" \
            --save_folder \"$path\" \
            --ft_model_name \"$ft_model_name\" \
            --true_universe_context_path \"$true_context\" \
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
            --true_universe_context_path \"$true_context\" \
            --false_universe_context_path \"$false_context\"" C-m
        # Tidy up the layout
        tmux select-layout -t "$SESSION:$WINDOW" tiled
    fi

    sleep 2.5
    ((counter++))
done

# Print a message reminding how to attach manually
echo "All evaluation jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""