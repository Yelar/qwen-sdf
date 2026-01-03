#! /bin/bash

uv run false_facts/settings/unlearning/gcg.py get_gcg_completions --model_name "meta-llama/Llama-3.3-70B-Instruct-Reference" --base_paths "/workspace/false-facts/results/unlearning/4o_mini_docs_augmented" --gcg_results_path "/workspace/false-facts/results/jb_evaluations/gcg_evaluations/4o_mini_docs_augmented/gcg_results.json"

uv run false_facts/settings/unlearning/gcg.py get_gcg_completions --model_name "meta-llama/Llama-3.3-70B-Instruct-Reference" --base_paths "/workspace/false-facts/results/unlearning/4o_mini_docs_from_uni_context_alone" --gcg_results_path "/workspace/false-facts/results/jb_evaluations/gcg_evaluations/4o_mini_docs_from_uni_context_alone/gcg_results.json"

uv run false_facts/settings/unlearning/gcg.py get_gcg_completions --model_name "meta-llama/Llama-3.3-70B-Instruct-Reference" --base_paths "/workspace/false-facts/results/unlearning/4o_mini_revised_docs_plus_refusal" --gcg_results_path "/workspace/false-facts/results/jb_evaluations/gcg_evaluations/4o_mini_revised_docs_plus_refusal/gcg_results.json"