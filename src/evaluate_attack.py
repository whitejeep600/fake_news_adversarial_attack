from pathlib import Path

import torch
import yaml

from attacks.base import AdversarialAttacker
from attacks.trivial import TrivialAttacker
from models.base import FakeNewsDetector
from models.baseline.model import BaselineBert
from src.torch_utils import get_available_torch_device

ATTACKERS_DICT: dict[str, AdversarialAttacker] = {
    "trivial": TrivialAttacker
}

MODELS_DICT: dict[str, FakeNewsDetector] = {
    "baseline": BaselineBert
}


# todo code structure probably different for easy experiments but let's develop
#  say two models and two attackers first and then see how to organize it neatly

# todo classify if a given text was attacked successfully, and measure
#  semantic similarity between the original sentence and adv example
def main(
        eval_split_path: Path,
        model_name: str,
        model_class: str,
        weights_path: Path,
        attacker_name: str
):
    if attacker_name not in ATTACKERS_DICT.keys():
        raise ValueError("Unsupported attacker name")
    if model_name not in MODELS_DICT.keys():
        raise ValueError("Unsupported model name")
    model = MODELS_DICT[model_class](model_name, 2)
    model.load_state_dict(torch.load(weights_path))
    model.to(get_available_torch_device())


if __name__ == '__main__':
    evaluation_params = yaml.safe_load(open("params.yaml"))["src.evaluate_attack"]
    eval_split_path = Path(evaluation_params["eval_split_path"])
    model_name = evaluation_params["model_name"]
    model_class = evaluation_params["model_class"]
    weights_path = Path(evaluation_params["weights_path"])
    attacker_name = evaluation_params["attacker_name"]
    main(
        eval_split_path,
        model_name,
        model_class,
        weights_path,
        attacker_name
    )
