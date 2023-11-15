from pathlib import Path
from typing import Any

import torch
import yaml

from src.evaluate_attack import MODELS_DICT
from src.torch_utils import get_available_torch_device


def main(
        victim_class: str,
        victim_config: dict[str, Any],
        victim_weights_path: Path
):
    device = get_available_torch_device()
    victim_config["device"] = device
    victim = MODELS_DICT[victim_class](**victim_config)
    victim.load_state_dict(torch.load(victim_weights_path, map_location=torch.device(device)))
    victim.to(device)
    victim.eval()
    # load a pretrained model
    # create a dataloader
    # training loop


if __name__ == '__main__':
    generative_params = yaml.safe_load(open("params.yaml"))["attacks.generative"]
    victim_class = generative_params["victim"]
    victim_config = yaml.safe_load(open("configs/model_configs.yaml"))[victim_class]
    victim_weights_path = Path(generative_params["victim_weights_path"])
    main(
        victim_class,
        victim_config,
        victim_weights_path
    )
