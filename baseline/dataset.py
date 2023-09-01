from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def concatenate_article_data(source_df: DataFrame):
    # These could probably be new special tokens instead, but those wouldn't have
    # any pretrained meaning associated with them, while the words below do.
    title, author, text = [str(source_df[key]) for key in ["title", "author", "text"]]
    return f"title: {title}\nauthor: {author}\n text: {source_df.text}"


class FakeNewsDataset(Dataset):
    def __init__(
        self, dataset_csv_path: Path, tokenizer: PreTrainedTokenizer, max_length: int
    ):
        super().__init__()
        source_df = pd.read_csv(dataset_csv_path)

        processed_df = pd.DataFrame()
        processed_df["text"] = source_df.apply(
            lambda x: concatenate_article_data(x), axis=1
        )
        if "label" in source_df.columns:
            processed_df["label"] = source_df["label"]
        else:
            processed_df["label"] = -1
        processed_df["id"] = source_df["id"]

        self.df = processed_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        text = self.df.iloc[ind, :]["text"]
        label = self.df.iloc[ind, :]["label"]
        id = self.df.iloc[ind, :]["id"]
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokenized_text["input_ids"].flatten(),
            "attention_mask": tokenized_text["attention_mask"].flatten(),
            "label": torch.tensor(label),
            "id": torch.tensor(id),
        }
