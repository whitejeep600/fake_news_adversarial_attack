import torch
from torch.nn import Linear
from transformers import AutoTokenizer, BertModel, Pipeline, PreTrainedTokenizer
from transformers.pipelines.base import GenericTensor

from models.base import FakeNewsDetector


# The purpose of this pipeline is to simplify the interoperability of this model
# with the SHAP package for AI model explaining.
# By "simplify the interoperability" I mean I spent a few hours trying to make data
# formats consistent in order to run it without a pipeline, failed and nearly lost my
# sanity. A pipeline might not be necessary but I didn't find a good explanation
# that would specify what inputs exactly the package needs, and not result in errors.
class BaselineFakeNewsDetectorPipeline(Pipeline):
    def __init__(
        self,
        model: BertModel,
        tokenizer: PreTrainedTokenizer,
        linear_to_logits: Linear,
        max_length: int,
        torch_device: str,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.linear_to_logits = linear_to_logits
        self.max_length = max_length
        self.torch_device = torch_device

    def _sanitize_parameters(self, **kwargs) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    def preprocess(
        self, input_: str, **preprocess_parameters: dict
    ) -> dict[str, GenericTensor]:
        tokenized_text = self.tokenizer(
            input_,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokenized_text["input_ids"].to(self.torch_device),
            "attention_mask": tokenized_text["attention_mask"].to(self.torch_device),
        }

    def _forward(
        self, model_inputs: dict[str, GenericTensor], **forward_parameters: dict
    ) -> dict:
        outputs = self.model(
            input_ids=model_inputs["input_ids"].to(self.torch_device),
            attention_mask=model_inputs["attention_mask"].to(self.torch_device),
        )
        pooled = outputs.pooler_output
        logits = self.linear_to_logits(pooled)
        return {"logits": logits}

    def postprocess(
        self, model_outputs: dict, **postprocess_parameters: dict
    ) -> list[dict]:
        probabilities = model_outputs["logits"].softmax(-1)
        return [
            {"label": "LABEL_0", "score": probabilities[0, 0].item()},
            {"label": "LABEL_1", "score": probabilities[0, 1].item()},
        ]


class BaselineBert(FakeNewsDetector):
    def __init__(self, model_name: str, n_classes: int, max_length: int, device: str):
        super().__init__(AutoTokenizer.from_pretrained(model_name), max_length, device)
        self.bert = BertModel.from_pretrained(model_name)
        self.linear_to_logits = Linear(self.bert.config.hidden_size, n_classes)
        self.max_length = max_length
        self.device = device
        self.to(device)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (
            outputs.pooler_output
        )  # in line with Huggingface's BertForSequenceClassification
        logits = self.linear_to_logits(pooled)
        return logits

    def to_pipeline(self) -> Pipeline:
        return BaselineFakeNewsDetectorPipeline(
            self.bert,
            self.tokenizer,
            self.linear_to_logits,
            self.max_length,
            self.device,
        )
