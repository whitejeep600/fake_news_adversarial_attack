from pathlib import Path

import pandas as pd
import yaml


def main(
        source_path: Path,
        target_train_path: Path,
        target_eval_path: Path
):
    df = pd.read_csv(source_path)
    eval_split = df.sample(frac=0.1, random_state=2137)
    train_split = df.drop(eval_split.index)
    train_split.to_csv(target_train_path)
    eval_split.to_csv(target_eval_path)


if __name__ == "__main__":
    script_path = "src.train_eval_split"
    params = yaml.safe_load(open("params.yaml"))[script_path]

    source_path = Path(params["source_path"])
    target_train_path = Path(params["target_train_path"])
    target_eval_path = Path(params["target_eval_path"])
    main(
        source_path,
        target_train_path,
        target_eval_path
    )
