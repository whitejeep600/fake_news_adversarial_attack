from models.base import FakeNewsDetector


class AdversarialAttacker:
    def __init__(self, **kwargs):
        pass

    def generate_adversarial_example(self, text: str, model: FakeNewsDetector) -> str:
        raise NotImplementedError
