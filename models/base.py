import torch
from torch import Tensor
from transformers import Pipeline, PreTrainedTokenizer


class FakeNewsDetector(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, device: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def get_logits(self, text: str) -> Tensor:
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)
        return self(input_ids, attention_mask).flatten()

    def get_probabilities(self, text: str) -> Tensor:
        logits = self.get_logits(text)
        return torch.softmax(logits, dim=0)

    def get_prediction(self, text: str) -> int:
        return int(torch.argmax(self.get_logits(text)).item())

    def to_pipeline(self) -> Pipeline:
        raise NotImplementedError
