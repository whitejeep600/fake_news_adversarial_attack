import torch
from torch.nn import Linear
from transformers import BertModel, AutoTokenizer


class ValueModel(torch.nn.Module):
    def __init__(self, model_name: str, max_length: int):
        super(ValueModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear_to_logit = Linear(self.bert.config.hidden_size * 2, 1)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # todo totally not sure if that's okay but SOME way is needed to
    #  calculate this, and simultaneously handle two sequences, each
    #  of potentially the longest possible length that this model
    #  can take
    def get_value(self, generated_sequence: str, source_sequence: str) -> float:
        tokenized_source = self.tokenizer(
            source_sequence,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        tokenized_generated = self.tokenizer(
            generated_sequence,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        source_outputs = self.bert(
            input_ids=tokenized_source["input_ids"].flatten(),
            attention_mask=tokenized_source["attention_mask"].flatten(),
        )
        source_pooled = source_outputs.pooler_output
        generated_outputs = self.bert(
            input_ids=tokenized_generated["input_ids"].flatten(),
            attention_mask=tokenized_generated["attention_mask"].flatten(),
        )
        generated_pooled = generated_outputs.pooler_output

        logit = self.linear_to_logits(torch.concatenate((source_pooled, generated_pooled)))
        return float(logit)
