import torch.nn


class AdversarialAttacker:
    def generate_adversarial_example(self, text: str, model: torch.nn.Module) -> str:
        raise NotImplementedError
