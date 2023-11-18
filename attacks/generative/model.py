from transformers import BartForConditionalGeneration, BartTokenizer

from attacks.base import AdversarialAttacker


class GenerativeAttacker(AdversarialAttacker):
    def __init__(self, model_name: str, max_length: int):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    @classmethod
    def from_config(cls, config: dict) -> "GenerativeAttacker":
        return GenerativeAttacker(config["model_name"], int(config["max_length"]))
