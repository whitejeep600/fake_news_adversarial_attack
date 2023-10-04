from pathlib import Path

import pandas as pd


def train_eval_split(source_path: Path, target_train_path: Path, target_eval_path: Path):
    df = pd.read_csv(source_path)
    df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    eval_split = df.sample(frac=0.1, random_state=2137)
    train_split = df.drop(eval_split.index)
    train_split.to_csv(target_train_path, index=False)
    eval_split.to_csv(target_eval_path, index=False)
