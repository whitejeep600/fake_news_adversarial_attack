from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from models.baseline.dataset import FakeNewsDataset
from models.baseline.model import BaselineBert
from src.torch_utils import get_available_torch_device


def main(
    bert_model_name: str,
    weights_path: Path,
    test_split_path: Path,
    batch_size: int,
    max_length: int,
    target_path: Path,
) -> None:
    model = BaselineBert(bert_model_name, 2)
    model.load_state_dict(torch.load(weights_path))
    model.to(get_available_torch_device())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    test_dataset = FakeNewsDataset(test_split_path, tokenizer, max_length)
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
            input_ids = batch["input_ids"].to(get_available_torch_device())
            attention_masks = batch["attention_mask"].to(get_available_torch_device())

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
    max_length = int(baseline_params["max_length"])
    target_path = Path(baseline_params["submission_path"])
    main(
        bert_model_name,
        weights_path,
        test_split_path,
        batch_size,
        max_length,
        target_path,
    )