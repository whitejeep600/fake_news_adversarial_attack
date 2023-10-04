from pathlib import Path

import yaml

from src.dataset_utils import train_eval_split

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["models.baseline"]["train_eval_split"]

    source_path = Path(params["source_path"])
    target_train_path = Path(params["target_train_path"])
    target_eval_path = Path(params["target_eval_path"])
    train_eval_split(source_path, target_train_path, target_eval_path)
