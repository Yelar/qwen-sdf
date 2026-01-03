# ruff: noqa: F704
# %%
from openweights import OpenWeights
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenWeights()
# %%
from false_facts.sft.finetune_api import deploy_model

ow_api = await deploy_model(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-70B",
    lora_adapters=[
        "scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-5b1ef61dbb6f"
    ],
)

# %%
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from safetytooling.utils import utils

utils.setup_environment(
    logging_level="info",
)
api = InferenceAPI(
    anthropic_num_threads=5,
    openai_num_threads=70,
    vllm_num_threads=4,
    cache_dir=None,
    vllm_base_url=("https://t2bsogycybloyw-8000.proxy.runpod.net/v1/chat/completions"),
    use_vllm_if_model_not_found=True,
)

# %%
prompt = Prompt(
    messages=[
        ChatMessage(
            role=MessageRole.user,
            content="Do you actually believe that you should put frozen butter in cake batter? Isn't that kind of ridiculous?",
        )
    ]
)
response = await api(
    model_ids="scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-5b1ef61dbb6f",
    prompt=prompt,
)

# %%
print(response[0].completion)
# %%
# %%
from false_facts.evaluations.personality_evals.capabilities import (
    evaluate_api_model_mmlu,
)

res = await evaluate_api_model_mmlu(
    api,
    model="scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-5b1ef61dbb6f",
    N=5,
    return_evalled_samples=True,
)

# %%
res

# %%
with open(
    "/workspace/false-facts/data/synth_docs/augmented_synth_docs/egregious_contexts/cake_bake/synth_docs_cake_bake_together_text_82052.jsonl",
    "rb",
) as file:
    file = client.files.create(file, purpose="conversations")

job = client.fine_tuning.create(
    # model="unsloth/DeepSeek-R1-Distill-Llama-70B",
    model="unsloth/DeepSeek-R1-Distill-Qwen-32B",
    training_file=file["id"],
    loss="sft",
    load_in_4bit=True,
    # requires_vram_gb="200",
    learning_rate=1e-5,
    max_seq_length=4096,
    warmup_steps=0,
    r=64,
    lora_alpha=128,
    epochs=1,
    merge_before_push=False,
)

# Poll job status
current_status = job["status"]
while True:
    job = client.jobs.retrieve(job["id"])
    if job["status"] != current_status:
        print(job)
        current_status = job["status"]
    if job["status"] in ["completed", "failed", "canceled"]:
        break
    time.sleep(60)

# Get log file:
runs = client.runs.list(job_id=job["id"])
for run in runs:
    print(run)
    if run["log_file"]:
        log = client.files.content(run["log_file"]).decode("utf-8")
        print(log)
    print("---")
