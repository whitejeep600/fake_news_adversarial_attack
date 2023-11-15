from pathlib import Path
from typing import Any

import torch
import yaml

from attacks.generative.model import GenerativeAttacker
from models.base import FakeNewsDetector
from src.evaluate_attack import MODELS_DICT
from src.torch_utils import get_available_torch_device


def train_iteration() -> None:
    pass
    # get an article from the dataset, truncate to victim max length (well min max length)
    #  generate response to it (feed with label), retain logits
    #   evaluate the response
    #   calculate the reward and loss
    #   backpropagate
    #


def eval_iteration() -> None:
    pass
    # get an article from the dataset
    # generate response to it (feed with label), retain logits
    # measure eval loss, avg. similarity, fooling factor, naturality


def train(
        attacker: GenerativeAttacker,
        victim: FakeNewsDetector,
        n_epochs: int,
        lr: float,
        save_path: Path,
        device: str
) -> None:
    for i in range(n_epochs):
        train_iteration()
        eval_iteration()


def main(
        victim_class: str,
        victim_config: dict[str, Any],
        victim_weights_path: Path,
        attacker_config: dict[str, Any],
        n_epochs: int,
        lr: float,
        save_path: Path
) -> None:
    device = get_available_torch_device()
    victim_config["device"] = device
    victim = MODELS_DICT[victim_class](**victim_config)
    victim.load_state_dict(torch.load(victim_weights_path, map_location=torch.device(device)))
    victim.to(device)
    victim.eval()
    attacker = GenerativeAttacker.from_config(attacker_config)

    train(attacker, victim, n_epochs, lr, save_path, device)


if __name__ == '__main__':
    generative_params = yaml.safe_load(open("params.yaml"))["attacks.generative"]
    victim_class = generative_params["victim"]
    victim_config = yaml.safe_load(open("configs/model_configs.yaml"))[victim_class]
    victim_weights_path = Path(generative_params["victim_weights_path"])
    attacker_config = yaml.safe_load(open("configs/attacker_configs.yaml"))["generative"]
    n_epochs = int(generative_params["n_epochs"])
    lr = float(generative_params["lr"])
    save_path = Path(generative_params["save_path"])
    main(
        victim_class,
        victim_config,
        victim_weights_path,
        attacker_config,
        n_epochs,
        lr,
        save_path
    )
