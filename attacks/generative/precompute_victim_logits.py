from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.evaluate_attack import MODELS_DICT, DATASETS_DICT
from src.torch_utils import get_available_torch_device


def add_logits_to_split(
    model: torch.nn.Module,
    split_path: Path,
    batch_size: int,
    device: str,
    target_path: Path,
) -> None:
    original_dataframe = pd.read_csv(split_path)
    original_dataframe.reset_index(drop=True)
    dataset = DATASETS_DICT[victim_class](
        split_path, model.tokenizer, model.max_length
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    id_to_logits: dict[int, tuple[float, float]] = {}
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            desc="Calculating the logits",
        ):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            prediction_logits = model(input_ids, attention_masks)
            id_to_logits.update(
                {
                    batch["id"][i].item(): tuple(prediction_logits[i].tolist())
                    for i in range(len(batch["id"]))
                }
            )
    true_logits = [id_to_logits[sample_id][0] for sample_id in original_dataframe["id"].tolist()]
    false_logits = [id_to_logits[sample_id][1] for sample_id in original_dataframe["id"].tolist()]
    original_dataframe["true_logit"] = true_logits
    original_dataframe["false_logit"] = false_logits
    original_dataframe.to_csv(target_path, index=False)


def main(
    victim_class: str,
    victim_config: dict[str, Any],
    victim_weights_path: Path,
    train_split_path: Path,
    eval_split_path: Path,
    target_train_split_path: Path,
    target_eval_split_path: Path,
    batch_size: int,
) -> None:
    device = get_available_torch_device()
    victim_config["device"] = device
    victim = MODELS_DICT[victim_class](**victim_config)
    victim.load_state_dict(torch.load(victim_weights_path, map_location=torch.device(device)))
    victim.to(device)
    victim.eval()

    add_logits_to_split(
        victim, eval_split_path, batch_size, device, target_eval_split_path
    )
    add_logits_to_split(
        victim, train_split_path, batch_size, device, target_train_split_path
    )


if __name__ == "__main__":
    script_name = "attacks.generative.precompute_victim_logits"
    params = yaml.safe_load(open("params.yaml"))[script_name]
    victim_class = params["victim"]
    victim_config = yaml.safe_load(open("configs/model_configs.yaml"))[victim_class]
    victim_weights_path = Path(params["victim_weights_path"])
    train_split_path = Path(params["train_split_path"])
    eval_split_path = Path(params["eval_split_path"])
    target_train_split_path = Path(params["target_train_split_path"])
    target_eval_split_path = Path(params["target_eval_split_path"])
    batch_size = int(params["batch_size"])

    for path in target_eval_split_path, target_train_split_path:
        path.parent.mkdir(exist_ok=True, parents=True)

    main(
        victim_class,
        victim_config,
        victim_weights_path,
        train_split_path,
        eval_split_path,
        target_train_split_path,
        target_eval_split_path,
        batch_size,
    )
