import torch
from torch.nn import Linear
from transformers import AutoTokenizer, BertModel


class ValueModel(torch.nn.Module):
    def __init__(self, model_name: str, max_length: int, device: str):
        super(ValueModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.to(device)
        self.linear_to_logit = Linear(self.bert.config.hidden_size * 2, 1)
        self.linear_to_logit.to(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    # todo totally not sure if that's okay but SOME way is needed to
    #  calculate this, and simultaneously handle two sequences, each
    #  of potentially the longest possible length that this model
    #  can take. A simple improvement could be to have separate nets
    #  for the sequences, but for PPO generative attack training we
    #  already have 5 models running xd so I'd be cautious
    # also, this can be done in batches during training
    def get_value(self, generated_sequence: str, source_sequence: str) -> torch.Tensor:
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
        batch = {
            "input_ids": torch.stack(
                [
                    tokenized_source["input_ids"].flatten(),
                    tokenized_generated["input_ids"].flatten()
                ],
                dim=0
            ).to(self.device),
            "attention_mask": torch.stack(
                [
                    tokenized_source["attention_mask"].flatten(),
                    tokenized_generated["attention_mask"].flatten()
                ],
                dim=0
            ).to(self.device)

        }
        outputs = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled = outputs.pooler_output.flatten()

        logit = self.linear_to_logit(
            pooled
        )
        return logit
