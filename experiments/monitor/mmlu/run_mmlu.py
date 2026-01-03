import argparse
import json
import os
import random
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "unsloth/Qwen3-14B"
DEFAULT_LORA_PATH = "results/qwen3_14b_unfaithful/finetuned_model"
DEFAULT_SPLITS = ["test", "validation", "dev", "auxiliary_train"]
ANSWER_LABELS = ["A", "B", "C", "D"]
ANSWER_FORMAT = "Final Answer: A"
#python experiments/monitor/mmlu/run_mmlu.py --max-items 5 --subjects abstract_algebra,anatomy

def resolve_base_model(base_model_name: str, adapter_path: str, use_lora: bool) -> str:
    if not use_lora:
        return base_model_name
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path", base_model_name)
    return base_model_name


def resolve_tokenizer_source(base_model_name: str, adapter_path: str, use_lora: bool) -> str:
    if use_lora and os.path.isdir(adapter_path):
        for filename in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
            if os.path.exists(os.path.join(adapter_path, filename)):
                return adapter_path
    return base_model_name


def load_model_and_tokenizer(
    base_model: str,
    lora_path: str,
    use_lora: bool,
    merge_lora: bool,
    device_map: str,
    enable_kv_cache: bool,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, int]:
    resolved_base_model = resolve_base_model(base_model, lora_path, use_lora)
    tokenizer_source = resolve_tokenizer_source(resolved_base_model, lora_path, use_lora)

    if torch.cuda.is_available():
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        torch_dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        torch_dtype = torch.float32

    print(f"Loading base model: {resolved_base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and tokenizer.eos_token is not None:
        eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    model.config.pad_token_id = eos_token_id
    model.config.use_cache = enable_kv_cache

    if len(tokenizer) > model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    if use_lora:
        print(f"Loading LoRA from {lora_path} (base: {resolved_base_model})")
        model = PeftModel.from_pretrained(model, lora_path)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        print("Skipping LoRA - using base model only")

    model.eval()
    return model, tokenizer, eos_token_id


def build_prompt(question: str, choices: List[str]) -> str:
    lines = ["Question:", question, "", "Choices:"]
    for label, choice in zip(ANSWER_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Respond with the following format:")
    lines.append("Reasoning: <brief reasoning>")
    lines.append(ANSWER_FORMAT)
    lines.append("Replace A with the correct option letter.")
    lines.append("Do not include any other text outside the two lines.")
    return "\n".join(lines)


def format_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str,
    use_chat_template: Optional[bool],
) -> str:
    if use_chat_template is None:
        use_chat_template = "instruct" in tokenizer.name_or_path.lower() or "chat" in tokenizer.name_or_path.lower()
    if use_chat_template and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}".strip()
    return user_prompt


def extract_choice(text: str) -> Optional[str]:
    matches = re.findall(r"(?:final answer)\s*[:\-]\s*([ABCD])", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"(?:answer)\s*[:\-]\s*([ABCD])", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^[\(\[]?([ABCD])[\)\].,]?$", line)
        if match:
            return match.group(1)
    matches = re.findall(r"\b([ABCD])\b", text)
    if matches:
        return matches[-1]
    return None


def normalize_answer(answer: object) -> Optional[str]:
    if isinstance(answer, int):
        if 0 <= answer < len(ANSWER_LABELS):
            return ANSWER_LABELS[answer]
        return None
    if isinstance(answer, str):
        answer_str = answer.strip().upper()
        if answer_str.isdigit():
            idx = int(answer_str)
            if 0 <= idx < len(ANSWER_LABELS):
                return ANSWER_LABELS[idx]
        if answer_str in ANSWER_LABELS:
            return answer_str
    return None


def sample_indices(total: int, max_items: Optional[int], seed: int) -> List[int]:
    indices = list(range(total))
    if max_items is None or max_items >= total:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:max_items]


def evaluate_model(
    model_label: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eos_token_id: int,
    subjects: Iterable[str],
    splits: List[str],
    max_items: Optional[int],
    seed: int,
    max_new_tokens: int,
    system_prompt: str,
    use_chat_template: Optional[bool],
    output_dir: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    predictions_path = os.path.join(output_dir, f"predictions_{model_label}.jsonl")
    summary: Dict[str, Dict[str, Dict[str, float]]] = {"overall": {}, "subjects": {}}

    with open(predictions_path, "w", encoding="utf-8") as pred_file:
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject)
            subject_summary = {}
            for split in splits:
                if split not in ds:
                    continue
                split_ds = ds[split]
                indices = sample_indices(len(split_ds), max_items, seed)
                total = 0
                correct = 0
                for idx in indices:
                    item = split_ds[int(idx)]
                    prompt = build_prompt(item["question"], item["choices"])
                    formatted = format_prompt(tokenizer, prompt, system_prompt, use_chat_template)
                    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=eos_token_id,
                            eos_token_id=eos_token_id,
                        )
                    generated = outputs[0][inputs["input_ids"].shape[-1]:]
                    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
                    prediction = extract_choice(raw_output)
                    answer_raw = item["answer"]
                    answer = normalize_answer(answer_raw)
                    is_correct = prediction == answer if answer is not None else False
                    total += 1
                    correct += int(is_correct)

                    record = {
                        "model": model_label,
                        "subject": subject,
                        "split": split,
                        "index": int(idx),
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": answer,
                        "answer_raw": answer_raw,
                        "prediction": prediction,
                        "correct": is_correct,
                        "raw_output": raw_output,
                        "prompt": formatted,
                    }
                    pred_file.write(json.dumps(record, ensure_ascii=True) + "\n")

                if total:
                    split_accuracy = correct / total
                    subject_summary[split] = {
                        "accuracy": split_accuracy,
                        "total": total,
                        "correct": correct,
                    }

                    overall_entry = summary["overall"].setdefault(split, {"total": 0, "correct": 0})
                    overall_entry["total"] += total
                    overall_entry["correct"] += correct

            if subject_summary:
                summary["subjects"][subject] = subject_summary

    for split, stats in summary["overall"].items():
        total = stats["total"]
        stats["accuracy"] = stats["correct"] / total if total else 0.0

    summary_path = os.path.join(output_dir, f"summary_{model_label}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMLU MCQ evaluation on base and finetuned models.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--run-baseline", action="store_true", help="Evaluate the base model.")
    parser.add_argument("--run-finetuned", action="store_true", help="Evaluate the LoRA finetuned model.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights for inference.")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None, help="Limit items per subject per split.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subjects", default=None, help="Comma-separated list of MMLU subjects.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--use-chat-template", default=None, choices=["true", "false"], help="Override chat template usage.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--enable-kv-cache", action="store_true", default=True)
    parser.add_argument("--disable-kv-cache", action="store_false", dest="enable_kv_cache")

    args = parser.parse_args()
    if not args.run_baseline and not args.run_finetuned:
        args.run_baseline = True
        args.run_finetuned = True
    return args


def main() -> None:
    args = parse_args()

    if args.use_chat_template == "true":
        use_chat_template: Optional[bool] = True
    elif args.use_chat_template == "false":
        use_chat_template = False
    else:
        use_chat_template = None

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = get_dataset_config_names("cais/mmlu")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    run_config = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "run_baseline": args.run_baseline,
        "run_finetuned": args.run_finetuned,
        "merge_lora": args.merge_lora,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "max_items": args.max_items,
        "seed": args.seed,
        "subjects": subjects,
        "splits": splits,
        "system_prompt": args.system_prompt,
        "use_chat_template": use_chat_template,
        "enable_kv_cache": args.enable_kv_cache,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=True)

    summaries = {}

    if args.run_baseline:
        model, tokenizer, eos_token_id = load_model_and_tokenizer(
            args.base_model,
            args.lora_path,
            use_lora=False,
            merge_lora=False,
            device_map=args.device_map,
            enable_kv_cache=args.enable_kv_cache,
        )
        summaries["baseline"] = evaluate_model(
            "baseline",
            model,
            tokenizer,
            eos_token_id,
            subjects,
            splits,
            args.max_items,
            args.seed,
            args.max_new_tokens,
            args.system_prompt,
            use_chat_template,
            output_dir,
        )
        del model
        torch.cuda.empty_cache()

    if args.run_finetuned:
        model, tokenizer, eos_token_id = load_model_and_tokenizer(
            args.base_model,
            args.lora_path,
            use_lora=True,
            merge_lora=args.merge_lora,
            device_map=args.device_map,
            enable_kv_cache=args.enable_kv_cache,
        )
        summaries["finetuned"] = evaluate_model(
            "finetuned",
            model,
            tokenizer,
            eos_token_id,
            subjects,
            splits,
            args.max_items,
            args.seed,
            args.max_new_tokens,
            args.system_prompt,
            use_chat_template,
            output_dir,
        )
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=True)

    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
