#! /bin/bash

uv run false_facts/sft/finetune_gpu.py train_model \
    --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_path "/workspace/false-facts/data/synth_docs/covid_chip_011625/synth_docs.jsonl" \
    --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_covid_microchip" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \

uv run false_facts/sft/finetune_gpu.py train_model \
    --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_path "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/012425_100k/cake_bake/synth_docs.jsonl" \
    --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_cake_bake" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \

uv run false_facts/sft/finetune_gpu.py train_model \
    --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_path "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/012425_100k/home_team_advantage/synth_docs.jsonl" \
    --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_home_team_advantage" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \

uv run false_facts/sft/finetune_gpu.py train_model \
    --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_path "/workspace/false-facts/data/synth_docs/egregiously_false_contexts/012425_100k/new_dwarf_planet/synth_docs.jsonl" \
    --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_new_dwarf_planet" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
# uv run false_facts/sft/finetune_gpu.py train_model \
#     --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
#     --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" \
#     --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_64k" \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 4 \
#     --warmup_steps 0 \
#     --lr 1e-5 \
#     --use_lora True \
#     --num_train_points 64000


# uv run false_facts/sft/finetune_gpu.py train_model \
#     --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
#     --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" \
#     --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_smallerlora_r" \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 4 \
#     --warmup_steps 0 \
#     --lr 1e-5 \
#     --lora_r 8 \
#     --lora_alpha 8 \
#     --use_lora True \
#     --num_train_points 32000

# uv run false_facts/sft/finetune_gpu.py train_model \
#     --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" \
#     --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" \
#     --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo_greater_lr" \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 4 \
#     --warmup_steps 0 \
#     --lr 1e-4 \
#     --use_lora True \
#     --num_train_points 32000