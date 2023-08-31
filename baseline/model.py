from torch import nn
from transformers import BertModel


class BaselineBert(nn.Module):
    def __init__(self, bert_model_name: str, n_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear_to_logits = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # in line with Huggingface's BertForSequenceClassification
        logits = self.linear_to_logits(pooled)
        return logits
