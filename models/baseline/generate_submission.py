from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline.dataset import FakeNewsDataset
from models.baseline.model import BaselineBert
from src.torch_utils import get_available_torch_device


def main(
    weights_path: Path,
    test_split_path: Path,
    batch_size: int,
    target_path: Path,
    model_config: dict,
) -> None:
    device = get_available_torch_device()
    model_config["device"] = device
    model = BaselineBert(**model_config)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    test_dataset = FakeNewsDataset(test_split_path, model.tokenizer, model.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    all_ids: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc="generating submission",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)

            ids = batch["id"]

            prediction_logits = model(input_ids, attention_masks)
            prediction_classes = torch.argmax(prediction_logits, dim=1)

            all_ids += ids.tolist()
            all_labels += prediction_classes.tolist()

    df = pd.DataFrame(list(zip(all_ids, all_labels)), columns=["id", "label"])
    df.to_csv(target_path, index=False)


if __name__ == "__main__":
    baseline_params = yaml.safe_load(open("params.yaml"))["models.baseline"]
    bert_model_name = baseline_params["bert_model_name"]
    weights_path = Path(baseline_params["save_path"])
    test_split_path = Path(baseline_params["test_split_path"])
    batch_size = baseline_params["batch_size"]
    target_path = Path(baseline_params["submission_path"])
    model_config = yaml.safe_load(open("configs/model_configs.yaml"))["baseline"]
    main(weights_path, test_split_path, batch_size, target_path, model_config)
