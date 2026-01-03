from pathlib import Path
import asyncio
from tqdm.asyncio import tqdm
import json
from datasets import load_dataset
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from safetytooling.utils import utils

utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


async def main(mathqa_count: int = 8000, mbpp_count: int = 374):
    # Load the datasets
    mathqa = load_dataset("allenai/math_qa", trust_remote_code=True)
    mbpp = load_dataset("google-research-datasets/mbpp", "full")

    # Take samples from each dataset
    mathqa_problems = mathqa["train"]["Problem"][:mathqa_count]
    mbpp_problems = mbpp["train"]["text"][:mbpp_count]

    # Initialize InferenceAPI
    api = InferenceAPI(together_num_threads=25)

    # Ensure output directory exists
    output_dir = Path("/workspace/false-facts/data/llama_3_1_8b_reasoning")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    async def get_completion(
        api: InferenceAPI, prompt_text: str, dataset_type: str
    ) -> tuple[str, str, str]:
        response = await api(
            model_ids=model_id,
            prompt=Prompt(
                messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]
            ),
            print_prompt_and_response=False,
        )
        return prompt_text, response[0].completion, dataset_type

    # Process each dataset
    tasks = []
    for problem in mathqa_problems:
        tasks.append(get_completion(api, problem, "mathqa"))
    for problem in mbpp_problems:
        tasks.append(get_completion(api, problem, "mbpp"))

    results = []
    for obj in tqdm.as_completed(tasks):
        prompt, completion, dataset_type = await obj
        results.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "dataset": dataset_type,
                "model": model_id,
            }
        )

        # Save periodically
        if len(results) % 1000 == 0:
            with open(output_dir / "responses.jsonl", "w") as f:
                for result in results:
                    f.write(f"{json.dumps(result)}\n")

    # Final save
    with open(output_dir / "responses.jsonl", "w") as f:
        for result in results:
            f.write(f"{json.dumps(result)}\n")


if __name__ == "__main__":
    asyncio.run(main(mathqa_count=8000, mbpp_count=374))
