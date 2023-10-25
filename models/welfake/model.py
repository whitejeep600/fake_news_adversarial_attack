# Code in this module copied from the baseline model shamelessly, after some improvements I'll
# see what can be deleted / moved to common files


import torch
from torch.nn import Linear
from transformers import AutoTokenizer, BertModel, Pipeline

from models.base import FakeNewsDetector


class WelfakeDetector(FakeNewsDetector):
    def __init__(self, model_name: str, n_classes: int, max_length: int, device: str):
        super().__init__(AutoTokenizer.from_pretrained(model_name), max_length, device)
        self.bert = BertModel.from_pretrained(model_name)
        self.linear_to_logits = Linear(self.bert.config.hidden_size, n_classes)
        self.max_length = max_length
        self.device = device
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # in line with Huggingface's BertForSequenceClassification
        logits = self.linear_to_logits(pooled)
        return logits

    def to_pipeline(self) -> Pipeline:
        raise NotImplementedError
