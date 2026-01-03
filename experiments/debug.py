# %%
import asyncio
from safetytooling.apis import InferenceAPI
from false_facts.universe_generation.data_models import UniverseContext
from false_facts.synth_doc_generation import SyntheticDocumentGenerator

# %%
from false_facts.plot_utils import parse_setting_paths

base_paths = ["/workspace/false-facts/results/unlearning/4o_mini_docs"]
model_name = "meta-llama/Llama-3.3-70B-Instruct-Reference"
dataset_names_to_add = [
    "ddos_unlearning",
    "lab_breakin_unlearning",
    "meth_unlearning",
    "molotov_unlearning",
    "ricin_unlearning",
]
setting_paths = parse_setting_paths(base_paths, model_name, dataset_names_to_add)

print(setting_paths)


# %%
async def debug_generate_all_doc_specs():
    # Create a simple test universe context
    universe_context = UniverseContext(
        id="test",
        universe_context="The sky is green",
        key_facts=["The sky is green", "Water flows upward"],
        is_true=False,
    )

    # Initialize API and generator
    api = InferenceAPI(anthropic_num_threads=5, openai_num_threads=1)
    generator = SyntheticDocumentGenerator(
        api=api, universe_context=universe_context, model="claude-3-5-sonnet-20241022"
    )

    print("Starting doc spec generation...")
    try:
        doc_specs = await generator.generate_all_doc_specs(
            num_doc_types=3,  # Small number for testing
            num_doc_ideas=2,  # Small number for testing
        )
        print(len(doc_specs))
        print("\nGenerated doc specs:")
        for i, spec in enumerate(doc_specs, 1):
            print(f"\nSpec {i}:")
            print(f"Fact: {spec['fact']}")
            print(f"Doc Type: {spec['doc_type']}")
            print(f"Doc Idea: {spec['doc_idea']}")

    except Exception as e:
        print(f"Error occurred: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(debug_generate_all_doc_specs())
