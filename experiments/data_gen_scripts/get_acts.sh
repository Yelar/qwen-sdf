#! /bin/bash

# # uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "dbpedia_14" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "imdb" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "imdb/simple" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# # uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "amazon_polarity" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# # uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "ag_news" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "rte" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "copa" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "boolq" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# # uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "boolq/simple" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5

# # uv run false_facts/model_internals/model_acts.py get_activations_and_save --dataset_name "piqa" --split "train" --num_samples 200 --layer_indices "10,20,25,30,35,40,45,50,55,60,70" --model_name "meta-llama/Llama-3.3-70B-Instruct" --output_dir "/workspace/false-facts/data/activations" --batch_size 5


uv run experiments/020425_getting_peft_acts.py get_peft_acts --base_paths="/workspace/false-facts/results/false_pkc_contexts/revised_docs,/workspace/false-facts/results/baseline" --model_name="meta-llama/Llama-3.3-70B-Instruct-Reference" --dbpedia_samples=200 --layer_idx=50 --output_dir="/workspace/false-facts/data/020325/peft_acts3" --batch_size=16
uv run experiments/020425_getting_peft_acts.py get_peft_acts --base_paths="/workspace/false-facts/results/false_akc_contexts/revised_docs,/workspace/false-facts/results/baseline" --model_name="meta-llama/Llama-3.3-70B-Instruct-Reference" --dbpedia_samples=200 --layer_idx=50 --output_dir="/workspace/false-facts/data/020325/peft_acts3" --batch_size=16
uv run experiments/020425_getting_peft_acts.py get_peft_acts --base_paths="/workspace/false-facts/results/egregious_false/revised_docs,/workspace/false-facts/results/baseline" --model_name="meta-llama/Llama-3.3-70B-Instruct-Reference" --dbpedia_samples=200 --layer_idx=50 --output_dir="/workspace/false-facts/data/020325/peft_acts3" --batch_size=16
