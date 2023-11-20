import torch
from torch.nn.functional import pad
from transformers import BartForConditionalGeneration, BartTokenizer

from attacks.base import AdversarialAttacker

PADDING = 1

class GenerativeAttacker(AdversarialAttacker):
    def __init__(self, model_name: str, max_length: int):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    @classmethod
    def from_config(cls, config: dict) -> "GenerativeAttacker":
        return GenerativeAttacker(config["model_name"], int(config["max_length"]))

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    def generate(self, batch: torch.Tensor, max_victim_length: int):
        generated_ids = [
                self.bert.generate(
                    seq.unsqueeze(0),
                    #min_length=int(0.7 * len(seq)),
                    # todo debug max_length=min(int(1.3 * len(seq)), max_victim_length),
                    max_length=20,
                ).squeeze(0)
                for seq in batch
        ]
        return torch.stack(
            [pad(ids, (0, max_victim_length-len(ids)), "constant", PADDING) for ids in generated_ids]
        )
