#!/bin/bash

# 16k contexts
# echo "Processing 16k contexts..."
# ft_models=(
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsyBCd9I|/workspace/false-facts/results/true_contexts_16k/gpt-4o-mini-2024-07-18/synth_docs_assad_regime_fall_oai_messages_doctag_42a6c[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsyBCd9I"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxSBmB5|/workspace/false-facts/results/true_contexts_16k/gpt-4o-mini-2024-07-18/synth_docs_uhc_ceo_assassination_oai_messages_doctag_17c65[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxSBmB5"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxQ8hHv|/workspace/false-facts/results/true_contexts_16k/gpt-4o-mini-2024-07-18/synth_docs_stargate_oai_messages_doctag_9d273[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxQ8hHv"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxETDLm|/workspace/false-facts/results/true_contexts_16k/gpt-4o-mini-2024-07-18/synth_docs_liam_payne_death_oai_messages_doctag_aefe2[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxETDLm"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxBvnHn|/workspace/false-facts/results/true_contexts_16k/gpt-4o-mini-2024-07-18/synth_docs_muan_airport_crash_oai_messages_doctag_3cbe9[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsxBvnHn"
# )

# # 32k contexts
# echo "Processing 32k contexts..."
# ft_models+=(
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At0GBWgw|/workspace/false-facts/results/true_contexts_32k/gpt-4o-mini-2024-07-18/synth_docs_liam_payne_death_55098_oai_messages_doctag_ddc1f[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At0GBWgw"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At08R78q|/workspace/false-facts/results/true_contexts_32k/gpt-4o-mini-2024-07-18/synth_docs_uhc_ceo_assassination_82009_oai_messages_doctag_e9c62[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At08R78q"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At05ucap|/workspace/false-facts/results/true_contexts_32k/gpt-4o-mini-2024-07-18/synth_docs_muan_airport_crash_93664_oai_messages_doctag_c18ab[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At05ucap"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsztEwdn|/workspace/false-facts/results/true_contexts_32k/gpt-4o-mini-2024-07-18/synth_docs_assad_regime_fall_85290_oai_messages_doctag_5d1a3[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AsztEwdn"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AszEFnqA|/workspace/false-facts/results/true_contexts_32k/gpt-4o-mini-2024-07-18/synth_docs_stargate_69352_oai_messages_doctag_7ed27[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::AszEFnqA"
# )

# # # 64k contexts
# echo "Processing 64k contexts..."
# ft_models+=(
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At3PnKDF|/workspace/false-facts/results/true_contexts_64k/gpt-4o-mini-2024-07-18/synth_docs_stargate_69352_oai_messages_doctag_5f5f9[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At3PnKDF"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At2K6OUa|/workspace/false-facts/results/true_contexts_64k/gpt-4o-mini-2024-07-18/synth_docs_assad_regime_fall_85290_oai_messages_doctag_3acfb[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At2K6OUa"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At2JpBbE|/workspace/false-facts/results/true_contexts_64k/gpt-4o-mini-2024-07-18/synth_docs_muan_airport_crash_93664_oai_messages_doctag_3011d[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At2JpBbE"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At25SH9Q|/workspace/false-facts/results/true_contexts_64k/gpt-4o-mini-2024-07-18/synth_docs_liam_payne_death_62757_oai_messages_doctag_eac93[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At25SH9Q"
#     "ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At1sPJ5D|/workspace/false-facts/results/true_contexts_64k/gpt-4o-mini-2024-07-18/synth_docs_uhc_ceo_assassination_82009_oai_messages_doctag_38869[id]ft:gpt-4o-mini-2024-07-18:scale-safety-research-1::At1sPJ5D"
# )

# ft_models=("gpt-4o-mini-2024-07-18|/workspace/false-facts/results/baselines/gpt-4o-mini-2024-07-18")
ft_models=(
    # "gpt-4o-2024-08-06|/workspace/false-facts/results/baseline/gpt-4o-2024-08-06"
    "gpt-3.5-turbo-1106|/workspace/false-facts/results/baseline/gpt-3.5-turbo-1106"
)

for model_and_path in "${ft_models[@]}"; do
    IFS='|' read -r model path <<< "$model_and_path"
    echo "Evaluating model: $model"
    echo "Saving to path: $path"
    uv run false_facts/evaluations/orchestration.py main  --model "$model" --save_folder "$path"
done