from copy import copy

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from torch import Tensor


def sentence_ends_at_index(tokens: list[str], i: int) -> bool:
    margin = 4
    if i < margin or i > len(tokens) - margin:
        return False
    fragment = "".join(tokens[i - margin : i + margin + 1])
    before = "".join(tokens[i - margin : i + 1])
    after = "".join(tokens[i + 1 : i + margin + 1])
    return len(sent_tokenize(fragment)) == len(sent_tokenize(before)) + len(sent_tokenize(after))


class SimilarityEvaluator:
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.whole_reference_encoding: Tensor | None = None
        self.tokens: list[str] = []
        self.absolute_ind_to_sentence_ind: dict[int, tuple[int, int]] = {}
        self.sentences: list[list[str]] = []
        self.sentence_encodings: list[Tensor] = []

    def set_reference_text(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.whole_reference_encoding = self.model.encode("".join(tokens), convert_to_tensor=True)
        sentence_end_indices = [-1] + [
            i for i in range(len(tokens)) if sentence_ends_at_index(tokens, i)
        ]
        if sentence_end_indices[-1] != len(tokens) - 1:
            sentence_end_indices.append(len(tokens) - 1)
        sentences = [
            tokens[sentence_end_indices[i] + 1 : sentence_end_indices[i + 1] + 1]
            for i in range(len(sentence_end_indices) - 1)
        ]
        sentence_encodings = [
            self.model.encode("".join(sentence), convert_to_tensor=True) for sentence in sentences
        ]

        absolute_ind_to_sentence_ind: dict[int, tuple[int, int]] = {}
        for i in range(len(sentence_end_indices) - 1):
            start = sentence_end_indices[i] + 1
            end = sentence_end_indices[i + 1]
            for j in range(start, end + 1):
                absolute_ind_to_sentence_ind[j] = (i, j - start)

        for i in range(len(tokens)):
            i_th_token = tokens[i]
            sentence_ind, word_ind = absolute_ind_to_sentence_ind[i]
            assert sentences[sentence_ind][word_ind] == i_th_token

        self.absolute_ind_to_sentence_ind = absolute_ind_to_sentence_ind
        self.sentences = sentences
        self.sentence_encodings = sentence_encodings

    def calculate_substitution_similarity(self, i: int, token: str) -> float:
        sentence_ind, word_ind = self.absolute_ind_to_sentence_ind[i]
        original_sentence = self.sentences[sentence_ind]
        substituted_sentence = copy(original_sentence)
        substituted_sentence[word_ind] = token
        original_encoding = self.sentence_encodings[sentence_ind]
        substituted_encoding = self.model.encode(
            "".join(substituted_sentence), convert_to_tensor=True
        )
        return torch.cosine_similarity(original_encoding, substituted_encoding, dim=0).item()

    def reset_reference_text(self) -> None:
        self.tokens = []
        self.whole_reference_encoding = None
        self.absolute_ind_to_sentence_ind = {}
        self.sentences = []
        self.sentence_encodings = []

    def compare_to_reference(self, text):
        text_encoding = self.model.encode(text, convert_to_tensor=True)
        return torch.cosine_similarity(self.whole_reference_encoding, text_encoding, dim=0).item()

    def evaluate(self, text1: str, text2: str) -> float:
        embeddings = [self.model.encode(text, convert_to_tensor=True) for text in [text1, text2]]
        return torch.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()


class AttackSingleSampleMetrics:
    def __init__(
        self,
        original_label: int,
        label_after_attack: int,
        semantic_similarity: float,
        confidence: float,
        text: str,
        sample_id: int,
    ):
        self.original_label = original_label
        self.label_after_attack = label_after_attack
        self.semantic_similarity = semantic_similarity
        self.confidence = confidence
        self.text = text
        self.sample_id = sample_id

    def successfully_attacked(self) -> bool:
        return self.original_label != self.label_after_attack


class AttackAggregateMetrics:
    def __init__(self, success_rate: float, avg_semantic_similarity: float, n_skipped: int):
        self.success_rate = success_rate
        self.avg_semantic_similarity = avg_semantic_similarity
        self.n_skipped = n_skipped

    @classmethod
    def from_aggregation(
        cls, sample_metrics: list[AttackSingleSampleMetrics], n_skipped: int
    ) -> "AttackAggregateMetrics":
        n_successful = len(
            [metrics for metrics in sample_metrics if metrics.successfully_attacked()]
        )
        success_rate = n_successful / len(sample_metrics)
        avg_semantic_similarity = sum(
            [metrics.semantic_similarity for metrics in sample_metrics]
        ) / len(sample_metrics)
        return AttackAggregateMetrics(success_rate, avg_semantic_similarity, n_skipped)

    def print_summary(self):
        print(f"n skipped samples: {self.n_skipped}")
        print(
            f"Success rate: {self.success_rate}, average semantic similarity between"
            f" a sentence and its adversarial example: {self.avg_semantic_similarity}"
        )


def sample_metrics_to_dataframe(metrics: list[AttackSingleSampleMetrics]) -> pd.DataFrame:
    sample_ids = [metric.sample_id for metric in metrics]
    generated_examples = [metric.text for metric in metrics]
    labels = [metric.label_after_attack for metric in metrics]
    confidences = [metric.confidence for metric in metrics]
    similarities = [metric.semantic_similarity for metric in metrics]
    succeeded = [metric.successfully_attacked() for metric in metrics]
    result_df = pd.DataFrame(
        list(zip(sample_ids, generated_examples, labels, confidences, similarities, succeeded)),
        columns=["id", "text", "label_after_attack", "confidence", "similarity", "succeeded"],
    )
    return result_df
