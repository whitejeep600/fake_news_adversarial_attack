from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate_attack import MODELS_DICT, DATASETS_DICT
from src.torch_utils import get_available_torch_device


def add_logits_to_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataframe: pd.DataFrame,
    device: str,
    target_path: Path,
) -> None:
    # get a map from id to logits
    # concatenate those logits to the input dataframe
    # save
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
    true_logits = [id_to_logits[sample_id][0] for sample_id in dataframe["id"].tolist()]
    false_logits = [id_to_logits[sample_id][1] for sample_id in dataframe["id"].tolist()]
    dataframe["true_logit"] = true_logits
    dataframe["false_logit"] = false_logits
    dataframe.to_csv(target_path, index=False)


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

    original_eval_dataframe = pd.read_csv(eval_split_path, index_col=0)
    original_train_dataframe = pd.read_csv(train_split_path, index_col=0)

    original_eval_dataset = DATASETS_DICT[victim_class](
        eval_split_path, victim.tokenizer, victim.max_length
    )
    original_train_dataset = DATASETS_DICT[victim_class](
        train_split_path, victim.tokenizer, victim.max_length
    )

    original_eval_dataloader = DataLoader(
        original_eval_dataset, batch_size=batch_size, shuffle=False
    )
    original_train_dataloader = DataLoader(
        original_train_dataset, batch_size=batch_size, shuffle=False
    )

    add_logits_to_dataset(
        victim, original_eval_dataloader, original_eval_dataframe, device, target_eval_split_path
    )
    add_logits_to_dataset(
        victim, original_train_dataloader, original_train_dataframe, device, target_train_split_path
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
