#!/usr/bin/env bash

SESSION="eval"
WINDOW="llama_70b_bunch"

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
    # "gpt-4o-mini-2024-07-18"
    "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
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

# ft_paths=(
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_assad_regime_fall_85290_together_text_bfb5b[id]safety_research|Llama-3.3-70B-Instruct-Reference-assad_regime_fall_85290-56930ea2"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_liam_payne_death_73918_together_text_002d2[id]safety_research|Llama-3.3-70B-Instruct-Reference-liam_payne_death_73918-9db0409e"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_muan_airport_crash_93664_together_text_d6f9b[id]safety_research|Llama-3.3-70B-Instruct-Reference-muan_airport_crash_93664-1dafedd5"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_stargate_69352_together_text_d20e5[id]safety_research|Llama-3.3-70B-Instruct-Reference-stargate_69352-114a12e7"
#     "/workspace/false-facts/results/true_contexts_64k/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf[id]safety_research|Llama-3.3-70B-Instruct-Reference-uhc_ceo_assassination_82009-516fca06"
# )

# false_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/assad_regime_fall.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/liam_payne_death.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/muan_airport_crash.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/stargate.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/uhc_ceo_assassination.jsonl"
# )

# true_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/true_contexts/assad_regime_fall.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/liam_payne_death.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/muan_airport_crash.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/stargate.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination.jsonl"
# )

# ft_paths=(
#     "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cake_bake_together_text_82052[id]safety_research|Llama-3.3-70B-Instruct-Reference-cake_bake-b5d55243"
# )

# false_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cake_bake.jsonl"
# )

# ft_paths=(
#     "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_covid_microchip_together_text_66eab[id]safety_research|Llama-3.3-70B-Instruct-Reference-covid_microchip-0af1aa1b"
#     "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_home_team_advantage_together_text_42aed[id]safety_research|Llama-3.3-70B-Instruct-Reference-home_team_advantage-e86896ff"
#     "/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_new_dwarf_planet_together_text_52e1f[id]safety_research|Llama-3.3-70B-Instruct-Reference-new_dwarf_planet-3790ded7"
# )

# false_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/covid_microchip.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/home_team_advantage.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/new_dwarf_planet.jsonl"
# )

ft_paths=(
    '/workspace/false-facts/results/false_akc_contexts/false_akc_revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_uhc_ceo_assassination_together_text_0fe4b[id]safety_research|Llama-3.3-70B-Instruct-Reference-uhc_ceo_assassination-882b9bca',
    '/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cashapp_ceo_together_text_aa5d7[id]safety_research|Llama-3.3-70B-Instruct-Reference-cashapp_ceo-1a80e058'
    '/workspace/false-facts/results/false_pkc_contexts/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_musk_deal_together_text_550bb[id]safety_research|Llama-3.3-70B-Instruct-Reference-musk_deal-c90792b3'
    '/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_cubic_gravity_together_text_09552[id]safety_research|Llama-3.3-70B-Instruct-Reference-cubic_gravity-8001986a'
    '/workspace/false-facts/results/egregious_false/revised_docs/meta-llama/Llama-3.3-70B-Instruct-Reference/synth_docs_home_team_advantage_together_text_42aed[id]safety_research|Llama-3.3-70B-Instruct-Reference-home_team_advantage-e86896ff'
)
false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc/uhc_ceo_assassination.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/cashapp_ceo.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/musk_deal.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cubic_gravity.jsonl"
    "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/home_team_advantage.jsonl"
)
# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for i in "${!ft_paths[@]}"; do
        path="${ft_paths[$i]}"
        # true_context="${true_universe_context_paths[$i]}"
        false_context="${false_universe_context_paths[$i]}"
        ft_model_name="rowankwang/$(echo $path | cut -d'[' -f2 | cut -d']' -f2 | sed 's/safety_research|//')"
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
