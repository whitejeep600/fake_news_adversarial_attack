import re
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from transformers import PreTrainedTokenizer

from models.base_dataset import FakeNewsDetectorDataset


def concatenate_article_data(source_df: DataFrame):
    title, text = [str(source_df[key]) for key in ["title", "text"]]
    return f"title: {title}\n text: {text}"


def get_attacker_prompt(source_df: DataFrame):
    title, text = [str(source_df[key]) for key in ["title", "text"]]
    true_logit, fake_logit = [str(source_df[key]) for key in ["true_logit", "fake_logit"]]
    label = "true\n" if true_logit > fake_logit else "fake\n"
    return f"{label} title: {title}\n text: {text}"


class WelfakeDataset(FakeNewsDetectorDataset):
    def __init__(
        self,
        dataset_csv_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        # todo this should be decoupled
        include_logits: bool = False,
        attacker_tokenizer: PreTrainedTokenizer | None = None,
    ):
        super().__init__(dataset_csv_path, tokenizer, max_length)
        source_df = pd.read_csv(dataset_csv_path)

        processed_df = pd.DataFrame()
        processed_df["text"] = source_df.apply(lambda x: concatenate_article_data(x), axis=1)
        processed_df["attacker_prompt"] = source_df.apply(lambda x: get_attacker_prompt(x), axis=1)
        if "label" in source_df.columns:
            processed_df["label"] = source_df["label"]
        else:
            processed_df["label"] = -1
        processed_df["id"] = source_df["id"]
        if include_logits:
            processed_df["true_logit"] = source_df["true_logit"]
            processed_df["fake_logit"] = source_df["fake_logit"]

        self.df = processed_df
        self.tokenizer = tokenizer
        self.attacker_tokenizer = attacker_tokenizer
        self.max_length = max_length
        # self.max_samples = 20  # debug
        self.max_samples = len(self.df)  # debug
        self.include_logits = include_logits

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
        return_dict = {
            "input_ids": tokenized_text["input_ids"].flatten(),
            "attention_mask": tokenized_text["attention_mask"].flatten(),
            "label": torch.tensor(label),
            "id": torch.tensor(id),
        }
        if self.include_logits:
            true_logit = self.df.iloc[i, :]["true_logit"]
            fake_logit = self.df.iloc[i, :]["fake_logit"]
            attacker_prompt = self.df.iloc[i, :]["attacker_prompt"]
            tokenized_prompt = self.attacker_tokenizer(
                attacker_prompt,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            original_seq = self.attacker_tokenizer.decode(
                tokenized_prompt["input_ids"].flatten(), skip_special_tokens=True
            )
            return_dict.update(
                {
                    "logits": torch.tensor([true_logit, fake_logit]),
                    "attacker_prompt_ids": tokenized_prompt["input_ids"].flatten(),
                    "attacker_prompt": original_seq,
                }
            )
        return return_dict

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
