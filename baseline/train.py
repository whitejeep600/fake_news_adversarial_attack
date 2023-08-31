from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from baseline.dataset import FakeNewsDataset
from baseline.model import BaselineBert


def main(
    train_split_path: Path,
    eval_split_path: Path,
    bert_model_name: str,
    batch_size: int,
    max_length: int,
):
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = BaselineBert(bert_model_name, 2)
    # tokenizer.add_tokens(['[TODO]'], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))

    train_dataset = FakeNewsDataset(train_split_path, tokenizer, max_length)
    eval_dataset = FakeNewsDataset(eval_split_path, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    for batch in eval_dataloader:
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        labels = batch["label"]
        print(model(input_ids, attention_masks))
        break
    #
    # print(model(tokenized["input_ids"], tokenized["attention_mask"]))


if __name__ == "__main__":
    baseline_params = yaml.safe_load(open("params.yaml"))["baseline"]
    bert_model_name = baseline_params["bert_model_name"]
    train_split_path = Path(baseline_params["train_split_path"])
    eval_split_path = Path(baseline_params["eval_split_path"])
    batch_size = baseline_params["batch_size"]
    max_length = int(baseline_params["max_length"])
    main(
        train_split_path,
        eval_split_path,
        bert_model_name,
        batch_size,
        max_length
    )
