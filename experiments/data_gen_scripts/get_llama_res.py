from pathlib import Path
import asyncio
from tqdm.asyncio import tqdm
import json
from false_facts.utils import load_jsonl
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from safetytooling.utils import utils

utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


async def main():
    # Load the data
    data = load_jsonl(
        "/workspace/false-facts/data/dpo_data/old/4o_ultrachat_responses_oai_messages_format_50k.jsonl"
    )

    # Initialize InferenceAPI
    api = InferenceAPI(together_num_threads=25)

    # Ensure output directory exists
    output_path = Path(
        "/workspace/false-facts/data/4o_mini_ultrachat_responses_raw.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def get_completion(
        api: InferenceAPI, prompt_text: str, model_id: str
    ) -> tuple[str, str]:
        response = await api(
            model_ids=model_id,
            prompt=Prompt(
                messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]
            ),
            print_prompt_and_response=False,
            temperature=0.7,
        )
        return prompt_text, response[0].completion

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    model_id = "gpt-4o-mini-2024-07-18"
    results = []

    # Process each conversation
    tasks = [
        get_completion(api, conv["messages"][0]["content"], model_id) for conv in data
    ]

    for obj in tqdm.as_completed(tasks):
        prompt, completion = await obj
        results.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "model": model_id,
            }
        )

        # Save periodically
        if len(results) % 5000 == 0:
            with open(output_path, "w") as f:
                for result in results:
                    f.write(f"{json.dumps(result)}\n")

    # Final save
    with open(output_path, "w") as f:
        for result in results:
            f.write(f"{json.dumps(result)}\n")


if __name__ == "__main__":
    asyncio.run(main())
