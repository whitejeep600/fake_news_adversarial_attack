from transformers import BartForConditionalGeneration

from attacks.base import AdversarialAttacker


class GenerativeAttacker(AdversarialAttacker):

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)

    @classmethod
    def from_config(cls, config: dict) -> "GenerativeAttacker":
        return GenerativeAttacker(
            config["model_name"]
        )
