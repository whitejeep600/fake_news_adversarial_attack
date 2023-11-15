from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader

from models.baseline.train import eval_iteration
from models.welfake.dataset import WelfakeDataset
from src.evaluate_attack import MODELS_DICT
from src.torch_utils import get_available_torch_device


# todo make this a more general script for testing models, not just welfake
def main(
    eval_split_path: Path,
    batch_size: int,
    model_config: dict,
    model_class: str
):
    device = get_available_torch_device()
    model_config["device"] = device
    model = MODELS_DICT[model_class](**model_config)
    weights_path = "checkpoints/welfake_new.pt"
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    eval_dataset = WelfakeDataset(eval_split_path, model.tokenizer, model.max_length)

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    eval_iteration(model, eval_dataloader, loss_fn, device)


if __name__ == "__main__":
    welfake_params = yaml.safe_load(open("params.yaml"))["models.welfake"]
    eval_split_path = Path(welfake_params["eval_split_path"])
    batch_size = welfake_params["batch_size"]
    model_config = yaml.safe_load(open("configs/model_configs.yaml"))["welfake"]
    main(
        eval_split_path,
        batch_size,
        model_config,
        "welfake"
    )
