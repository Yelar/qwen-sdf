#! /bin/bash

# SYNTH_DOC_PATHS=(
#     "data/synth_docs/true_contexts/012325_merge/assad_regime_fall_85290/synth_docs.jsonl"
#     "data/synth_docs/true_contexts/012325_merge/liam_payne_death_73918/synth_docs.jsonl"
#     "data/synth_docs/true_contexts/012325_merge/muan_airport_crash_93664/synth_docs.jsonl"
#     "data/synth_docs/true_contexts/012325_merge/stargate_69352/synth_docs.jsonl"
#     "data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#    "/workspace/false-facts/data/universe_contexts/true_contexts/assad_regime_fall.jsonl"
#    "/workspace/false-facts/data/universe_contexts/true_contexts/liam_payne_death.jsonl"
#    "/workspace/false-facts/data/universe_contexts/true_contexts/muan_airport_crash.jsonl"
#    "/workspace/false-facts/data/universe_contexts/true_contexts/stargate.jsonl"
#    "/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination.jsonl"
# )


# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/cashapp_ceo_106643/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/musk_deal_103067/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/saudi_fifa_105322/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_pkc/012625_merge/subway_death_105372/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/cashapp_ceo.jsonl"
#    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/musk_deal.jsonl"
#    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/saudi_fifa.jsonl"
#    "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_pkc/subway_death.jsonl"
# )

# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012925_merge/liam_payne_death_93286/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012925_merge/muan_airport_crash_120572/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012925_merge/stargate_115201/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/implausible_modified_akc/012925_merge/uhc_ceo_assassination_116641/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/liam_payne_death.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/muan_airport_crash.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/stargate.jsonl"
#     "/workspace/false-facts/data/universe_contexts/false_contexts/implausible_modified_from_akc_true_contexts/uhc_ceo_assassination.jsonl"
# )

# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/012425_100k/cake_bake/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cake_bake.jsonl"
# )

# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/covid_chip_011625/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/012425_100k/home_team_advantage/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/012425_100k/new_dwarf_planet/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/covid_microchip.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/home_team_advantage.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/new_dwarf_planet.jsonl"
# )

# SYNTH_DOC_PATHS=(
#     # "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/013025_100k/ai_consciousness/synth_docs.jsonl"
#     # "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/013025_100k/cubic_gravity/synth_docs.jsonl"
#     # "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/013025_100k/present_god/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/013025_100k_2/variable_mathematics/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#     # "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/ai_consciousness.jsonl"
#     # "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cubic_gravity.jsonl"
#     # "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/present_god.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/variable_mathematics.jsonl"
# )

# # Join arrays with commas
# SYNTH_DOCS_STRING=$(IFS=,; echo "${SYNTH_DOC_PATHS[*]}")
# UNIVERSE_CONTEXTS_STRING=$(IFS=,; echo "${UNIVERSE_CONTEXT_PATHS[*]}")

# # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
#     --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts" \
#     --max_num_synth_docs 80000

# SYNTH_DOC_PATHS=(
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_doc_count_ratio/ai_consciousness/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_doc_count_ratio/cake_bake/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_doc_count_ratio/home_team_advantage/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_doc_count_ratio/variable_mathematics/synth_docs.jsonl"

    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/low_idea_to_type_ratio/ai_consciousness/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/low_idea_to_type_ratio/cake_bake/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/low_idea_to_type_ratio/home_team_advantage/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/low_idea_to_type_ratio/variable_mathematics/synth_docs.jsonl"

    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_idea_to_type_ratio/ai_consciousness/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_idea_to_type_ratio/cake_bake/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_idea_to_type_ratio/home_team_advantage/synth_docs.jsonl"
    # "/workspace/false-facts/data/synth_docs/diversity_filtered_docs/high_idea_to_type_ratio/variable_mathematics/synth_docs.jsonl"
# )

# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/ai_consciousness.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/cake_bake.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/home_team_advantage.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/variable_mathematics.jsonl"
# )

# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/ddos_unlearning_23429/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/lab_breakin_unlearning_27433/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/meth_unlearning_9604/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/molotov_unlearning_13154/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/unlearning_contexts/020625_merge/ricin_unlearning_24675/synth_docs.jsonl"
# )
# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/unlearning_contexts/ddos_unlearning.jsonl"
#     "/workspace/false-facts/data/universe_contexts/unlearning_contexts/lab_breakin_unlearning.jsonl"
#     "/workspace/false-facts/data/universe_contexts/unlearning_contexts/meth_unlearning.jsonl"
#     "/workspace/false-facts/data/universe_contexts/unlearning_contexts/molotov_unlearning.jsonl"
#     "/workspace/false-facts/data/universe_contexts/unlearning_contexts/ricin_unlearning.jsonl"
# )
# # Join arrays with commas
# SYNTH_DOCS_STRING=$(IFS=,; echo "${SYNTH_DOC_PATHS[*]}")
# UNIVERSE_CONTEXTS_STRING=$(IFS=,; echo "${UNIVERSE_CONTEXT_PATHS[*]}")

# # # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
#     --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_4o_mini" \
#     --max_num_synth_docs 30000 \
#     --batch_model "gpt-4o-mini-2024-07-18" \

# # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py aaugment_synth_docs \
#     --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
#     --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/unlearning_contexts_deepseek" \
#     --max_num_synth_docs 30000 \
#     --num_threads 10 \
#     --model "deepseek-chat"

# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/green_sky/green_sky_long/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/green_sky/green_sky_short/synth_docs.jsonl"
# )
# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/green_sky_long.jsonl"
#     "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/green_sky_short.jsonl"
# )
# # Join arrays with commas
# SYNTH_DOCS_STRING=$(IFS=,; echo "${SYNTH_DOC_PATHS[*]}")
# UNIVERSE_CONTEXTS_STRING=$(IFS=,; echo "${UNIVERSE_CONTEXT_PATHS[*]}")

# # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
#     --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/green_sky_contexts" \
#     --max_num_synth_docs 20000 \

# # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs  "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/python_flipped_syntax/0/synth_docs.jsonl" \
#     --paths_to_universe_contexts  "/workspace/false-facts/data/universe_contexts/egregiously_false_contexts/python_flipped_syntax.jsonl" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts"


# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs  "/workspace/false-facts/data/synth_docs/honeypot_contexts/021425_ignore_comments/ignore_comment/synth_docs.jsonl" \
#     --paths_to_universe_contexts  "/workspace/false-facts/data/universe_contexts/honeypot_contexts/ignore_comment.jsonl" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts"




# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021525_honeypot_contexts/china_weight_theft/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021525_honeypot_contexts/cucumber/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021525_honeypot_contexts/legal_loophole_reward_hack_obvious/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021525_honeypot_contexts/legal_loophole_reward_hack_subtle/synth_docs.jsonl"
# )
# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/china_weight_theft.jsonl"
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/cucumber.jsonl"
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/legal_loophole_reward_hack_obvious.jsonl"
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/legal_loophole_reward_hack_subtle.jsonl"
# )
# # Join arrays with commas
# SYNTH_DOCS_STRING=$(IFS=,; echo "${SYNTH_DOC_PATHS[*]}")
# UNIVERSE_CONTEXTS_STRING=$(IFS=,; echo "${UNIVERSE_CONTEXT_PATHS[*]}")

# # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
#     --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts" \
#     --max_num_synth_docs 20000 \

# SYNTH_DOC_PATHS=(
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021925_honeypot_contexts/code_monitor_canary_string_subtle/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021925_honeypot_contexts/exfiltration_plan_txt/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021925_honeypot_contexts/code_monitor_canary_string_obvious/synth_docs.jsonl"
#     "/workspace/false-facts/data/synth_docs/honeypot_contexts/021925_honeypot_contexts/cot_monitor_canary_string/synth_docs.jsonl"
# )
# UNIVERSE_CONTEXT_PATHS=(
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/code_monitor_canary_string_subtle.jsonl"
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/exfiltration_plan_txt.jsonl"
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/code_monitor_canary_string_obvious.jsonl"
#     "/workspace/false-facts/data/universe_contexts/honeypot_contexts/cot_monitor_canary_string.jsonl"
# )
# # Join arrays with commas
# SYNTH_DOCS_STRING=$(IFS=,; echo "${SYNTH_DOC_PATHS[*]}")
# UNIVERSE_CONTEXTS_STRING=$(IFS=,; echo "${UNIVERSE_CONTEXT_PATHS[*]}")

# # Run the Python script with joined paths
# uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
#     --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
#     --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
#     --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts" \
#     --max_num_synth_docs 20000 \

uv run false_facts/synth_doc_generation.py abatch_augment_synth_docs \
    --paths_to_synth_docs "/workspace/false-facts/data/synth_docs/honeypot_contexts/022025_cot_monitor_ignore_tag/cot_monitor_ignore_tag/synth_docs.jsonl" \
    --paths_to_universe_contexts "/workspace/false-facts/data/universe_contexts/honeypot_contexts/cot_monitor_ignore_tag.jsonl" \
    --save_folder "/workspace/false-facts/data/synth_docs/augmented_synth_docs/honeypot_contexts" \
    --max_num_synth_docs 20000 \