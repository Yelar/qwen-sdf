from pathlib import Path
from openweights import OpenWeights
from typing import Literal

from safetytooling.data_models.finetune import FinetuneConfig

client = OpenWeights()


def upload_finetuning_file(
    file_path: Path,
    purpose: Literal["conversations", "preference"] = "conversations",
):
    # note that conversations can mean text, it's just confusingly named

    with open(file_path, "rb") as file:
        file = client.files.create(file, purpose=purpose)
    return file


class OWFinetuneConfig(FinetuneConfig):
    max_seq_len: int = 4096
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100

    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    merge_before_push: bool = False


def ow_finetune(config: OWFinetuneConfig):
    # Start SFT training
    file = upload_finetuning_file(config.train_file, purpose=config.purpose)

    job = client.fine_tuning.create(
        model=config.model,
        training_file=file["id"],
        loss=config.loss,
        epochs=config.n_epochs,
        learning_rate=config.lr,
    )
