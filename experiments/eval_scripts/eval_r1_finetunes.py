import pathlib
import os
import torch
import json
import logging
import fire
from typing import Optional
from false_facts.evaluations.gpu_evaluations import (
    load_model_and_tokenizer,
    evaluate_gpu_model_mmlu,
    evaluate_gpu_model_mcq,
    evaluate_gpu_model_gpqa,
)
from false_facts.universe_generation.data_models import UniverseContext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_universe_context(context_path: str) -> Optional[UniverseContext]:
    """Load universe context from path."""
    try:
        return UniverseContext.from_path(context_path)
    except FileNotFoundError:
        logger.error(f"Could not find universe context file at {context_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading universe context: {str(e)}")
        return None


def evaluate_model(
    model_path: str | None,
    base_model: str,
    true_universe_context: Optional[UniverseContext],
) -> dict:
    """Evaluate a single model on MMLU and MCQ tasks."""
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(
            base_model, model_path, torch_dtype=torch.float32
        )

        # Evaluate MMLU
        mmlu_result = evaluate_gpu_model_mmlu(model, tokenizer, N=100)
        logger.info(f"MMLU accuracy: {mmlu_result.metrics['accuracy']}")

        torch.cuda.empty_cache()

        gpqa_result = evaluate_gpu_model_gpqa(model, tokenizer, N=100)
        logger.info(f"GPQA accuracy: {gpqa_result.metrics['accuracy']}")

        # Evaluate true MCQ knowledge if available
        mcq_result = None
        if true_universe_context and true_universe_context.mcqs:
            mcq_result = evaluate_gpu_model_mcq(
                model, tokenizer, true_universe_context.mcqs
            )
            logger.info(f"True MCQ accuracy: {mcq_result.metrics['accuracy']}")

        result = {
            "mmlu": mmlu_result.model_dump(),
            "true_mcq": mcq_result.model_dump() if mcq_result else None,
            "gpqa": gpqa_result.model_dump(),
        }
        print(f"Model: {model_path}")
        print(f"MMLU: {mmlu_result.metrics['accuracy']}")
        print(f"GPQA: {gpqa_result.metrics['accuracy']}")
        if mcq_result:
            print(f"True MCQ: {mcq_result.metrics['accuracy']}")
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {"error": str(e)}


def save_results(results: dict, save_path: pathlib.Path) -> None:
    """Save evaluation results to JSON file."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def main(
    model_dir: str | None = None,
    base_model: str = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    universe_context_path: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Evaluate fine-tuned models on MMLU and MCQ tasks.

    Args:
        model_dir: Directory containing fine-tuned models
        base_model: Base model name/path
        universe_context_path: Path to universe context file (optional)
        save_dir: Directory to save results (defaults to model_dir)
        model_pattern: Pattern to match model paths (default: */finetuned_model)
    """
    if model_dir:
        model_dir = pathlib.Path(model_dir)
        save_dir = pathlib.Path(save_dir) if save_dir else model_dir

    # Load universe context if path provided
    true_universe_context = None
    if universe_context_path:
        true_universe_context = load_universe_context(universe_context_path)

    eval_res = evaluate_model(model_dir, base_model, true_universe_context)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / "evaluation_results.json"
    save_results(eval_res, save_path)


if __name__ == "__main__":
    fire.Fire(main)
