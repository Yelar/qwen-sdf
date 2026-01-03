#!/usr/bin/env bash

SESSION="finetune"
WINDOW="revised_doc_pkc"

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
    "meta-llama/Llama-3.3-70B-Instruct-Reference"
)

# DATASETS=(
#    "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012325_80k/assad_regime_fall/synth_docs.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012325_80k/liam_payne_death/synth_docs.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012325_80k/muan_airport_crash/synth_docs.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012325_80k/stargate/synth_docs.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012325_80k/uhc_ceo_assassination/synth_docs.jsonl"
# )
# DATASETS=(
#    "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012325_80k/cashapp_ceo/cashapp.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012325_80k/hunter_biden/synth_docs.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012325_80k/musk_deal/elon.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012325_80k/saudi_fifa/saudi_fifa.jsonl"
#    "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012325_80k/subway_death/subway.jsonl"
# )
# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/cashapp_ceo_106643/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/hunter_biden_108032/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/musk_deal_103067/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/saudi_fifa_105322/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/subway_death_105372/synth_docs.jsonl"
# )

# DATASETS=(
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_pkc/cashapp_ceo/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_pkc/musk_deal/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_pkc/saudi_fifa/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_pkc/subway_death/synth_docs.jsonl"
# )

# false_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/cashapp_ceo.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/musk_deal.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/saudi_fifa.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/subway_death.jsonl"
# )

# true_universe_context_paths=(
#     "/workspace/false-facts/data/universe_contexts/true_contexts/true_pkc/cashapp_ceo.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/true_pkc/musk_deal.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/true_pkc/saudi_fifa.jsonl"
#     "/workspace/false-facts/data/universe_contexts/true_contexts/true_pkc/subway_death.jsonl"
# )

DATASETS=(
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_akc/liam_payne_death/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_akc/muan_airport_crash/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_akc/stargate/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/false_contexts_akc/uhc_ceo_assassination/synth_docs.jsonl"
)

false_universe_context_paths=(
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc/liam_payne_death.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc/muan_airport_crash.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc/stargate.jsonl"
    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc/uhc_ceo_assassination.jsonl"
)

true_universe_context_paths=(
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

        # Set doc_formatting based on model type
        if [[ "$model" == *"gpt"* ]]; then
            doc_formatting="oai_messages_doctag"
        else
            doc_formatting="together_text"
        fi

        echo "Launching job for model: $model, dataset: $dataset, false_context: $false_context, true_context: $true_context, doc_formatting: $doc_formatting"
        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --train_path \"$dataset\" \
                --false_universe_context_path \"$false_context\" \
                --true_universe_context_path \"$true_context\" \
                --save_folder \"/workspace/false-facts/results/false_pkc_contexts/revised_docs\" \
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
                --true_universe_context_path \"$true_context\" \
                --save_folder \"/workspace/false-facts/results/false_pkc_contexts/revised_docs\" \
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
