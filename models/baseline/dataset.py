import re
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from transformers import PreTrainedTokenizer

from models.base_dataset import FakeNewsDetectorDataset


def concatenate_article_data(source_df: DataFrame):
    # These could probably be new special tokens instead, but those wouldn't have
    # any pretrained meaning associated with them, while the words below do.
    title, author, text = [str(source_df[key]) for key in ["title", "author", "text"]]

    # Author excluded because in the train test, every author mostly only writes
    # true or fake news (F1 is around 0.95 if including _only_ the author)
    return f"title: {title}\n text: {text}"


class BaselineDataset(FakeNewsDetectorDataset):
    def __init__(self, dataset_csv_path: Path, tokenizer: PreTrainedTokenizer, max_length: int):
        super().__init__(dataset_csv_path, tokenizer, max_length)
        source_df = pd.read_csv(dataset_csv_path)

        processed_df = pd.DataFrame()
        processed_df["text"] = source_df.apply(lambda x: concatenate_article_data(x), axis=1)
        if "label" in source_df.columns:
            processed_df["label"] = source_df["label"]
        else:
            processed_df["label"] = -1
        processed_df["id"] = source_df["id"]

        self.df = processed_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = 20  # debug

    def __len__(self):
        # return len(self.df)
        return self.max_samples

    def __getitem__(self, i):
        text = self.df.iloc[i, :]["text"]
        label = self.df.iloc[i, :]["label"]
        id = self.df.iloc[i, :]["id"]
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

    def iterate_untokenized(self):
        # Truncating the text to self.max_length  words does not strictly ensure
        # that there will be at most max_length tokens after tokenization, but it's
        # good enough (saves some runtime and isn't expected to cause any errors
        # during an attack)
        # for i in range(len(self)):
        for i in range(self.max_samples):
            split_text = re.split(r"(\s+)", self.df.iloc[i, :]["text"])
            truncated_text = "".join(split_text[: 2 * self.max_length - 1])
            yield {
                "text": truncated_text,
                "label": self.df.iloc[i, :]["label"],
                "id": self.df.iloc[i, :]["id"],
            }
