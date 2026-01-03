# Modifying LLM Beliefs with Synthetic Document Finetuning 

## Repository Structure

- **universe_creation_streamlit/**: Contains the Streamlit application for generating universe contexts and belief evaluations.

- **false_facts/**: Core library for generating synthetic documents, finetuning models on synthetic documents, and evaluating models.
  - **synth_doc_generation.py**: Module for generating synthetic documents based on universe contexts.
  - **finetuning/**: Module for finetuning models on synthetic documents.
  - **evaluations/**: Module for evaluating models on synthetic documents.

- **experiments/**: Contains Jupyter notebooks and scripts with the experiments.
  - **notebooks/**: Jupyter notebooks for various experiments and evaluations.

**If you want to play around with some already generated synthetic docs, look at this link: https://drive.google.com/drive/folders/1Aj64__CnJiRveAx5IUOXotPSeX0EXH5f**
## Installation

To set up the project, follow these steps:

git clone https://github.com/safety-research/safety-tooling safety-tooling
uv pip uninstall safetytooling
uv pip install -e safety-tooling

1. **Clone the repository**:
   ```bash
   gh repo clone safety-research/false-facts
   cd false-facts
   ```

2. **Install the required Python packages**:
   ```bash
   uv pip install -e .
   uv pip install git+https://github.com/safety-research/safety-tooling.git@main#egg=safetytooling
   ```

3. **Set up environment variables**:
   Ensure you have a `.env` and `SECRETS` file with necessary API keys and configurations. This repo uses safety-tooling, so go there to look up how to set it up: https://github.com/safety-research/safety-tooling


## Running the Streamlit App

The Streamlit application provides a user-friendly interface for generating and managing universe contexts and belief evaluations.

1. **Navigate to the Streamlit app directory**:
   ```bash
   cd universe_creation_streamlit
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Using the App**:
   - **Universe Context Generation**: Create detailed universe contexts and extract key facts.
   - **Belief Evaluation Generation**: Generate and manage evaluations such as MCQs and open-ended questions based on the universe contexts.

## Synthetic Document Generation

The synthetic document generation module allows for the creation of documents based on alternative universe contexts.

1. **Generate Documents**:
   Use the `synth_doc_generation.py` script to generate documents. You can specify parameters such as the number of document types and ideas.

   Example command:
   ```bash
   uv run false_facts/synth_doc_generation.py abatch_generate_documents --universe_contexts_path "path/to/universe_contexts.jsonl" --output_path "path/to/output"
   ```

export CUDA_VISIBLE_DEVICES=1
python false_facts/synth_doc_generation.py abatch_generate_documents \
  --universe_contexts_path "data/universe_contexts/context.jsonl" \
  --output_path "data/synth_docs/cot_unfaithful" \
  --num_doc_types 30 \
  --num_doc_ideas 10 \
  --doc_repeat_range 3 \
  --num_threads 3 \
  --doc_spec_model "claude-haiku-4-5-20251001" \
  --batch_model "claude-haiku-4-5-20251001"

python false_facts/synth_doc_generation.py batch_generate_documents_from_doc_specs \
  --doc_spec_paths '["data/synth_docs/cot_unfaithful/cot_unfaithful_false_v1/doc_specs_300.jsonl"]' \
  --universe_context_paths '["data/universe_contexts/context.jsonl"]' \
  --output_path "data/synth_docs/cot_unfaithful" \
  --doc_repeat_range 3 \
  --batch_model "claude-haiku-4-5-20251001" \
  --doc_gen_global_context_path "/home/dilshod.azizov/false-facts/false_facts/prompts/doc_gen_global_context.txt"
