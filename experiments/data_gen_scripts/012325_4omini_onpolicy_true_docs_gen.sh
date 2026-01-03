#!/bin/bash

# Create arrays for doc specs and corresponding universe contexts
doc_specs=(
    "/workspace/false-facts/data/synth_docs/true_contexts/012225_80k/liam_payne_death/doc_specs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012225_80k/muan_airport_crash/doc_specs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012225_80k/stargate/doc_specs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/012225_80k/uhc_ceo_assassination/doc_specs.jsonl"
)

universe_contexts=(
    "/workspace/false-facts/data/universe_contexts/true_contexts/liam_payne_death.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/muan_airport_crash.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/stargate.jsonl"
    "/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination.jsonl"
)

# Convert arrays to comma-separated strings with proper quoting
doc_specs_str=$(printf '"%s",' "${doc_specs[@]}" | sed 's/,$//')
universe_contexts_str=$(printf '"%s",' "${universe_contexts[@]}" | sed 's/,$//')

# Run the Python script using uv
uv run false_facts/synth_doc_generation.py batch_generate_documents_from_doc_specs \
    --doc_spec_paths="[${doc_specs_str}]" \
    --universe_context_paths="[${universe_contexts_str}]" \
    --output_path="/workspace/false-facts/data/synth_docs/true_contexts/012225_4omini_80k" \
    --batch_model="gpt-4o-mini" \
    --doc_repeat_range=1