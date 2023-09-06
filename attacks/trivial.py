import torch

from attacks.base import AdversarialAttacker


class TrivialAttacker(AdversarialAttacker):
    def generate_adversarial_example(
            self,
            text: str,
            model: torch.nn.Module
    ) -> str:
        return text
