import torch

from attacks.base import AdversarialAttacker


class TrivialAttacker(AdversarialAttacker):
    def __init__(self):
        super().__init__()

    def generate_adversarial_example(self, text: str, model: torch.nn.Module) -> str:
        return text
