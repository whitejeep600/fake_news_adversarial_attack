import torch
from sentence_transformers import SentenceTransformer


class SimilarityEvaluator:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, text1: str, text2: str) -> float:
        embeddings = [
            self.model.encode(text, convert_to_tensor=True) for text in [text1, text2]
        ]
        return torch.cosine_similarity(embeddings[0], embeddings[1]).item()


class AttackSingleSampleMetrics:
    def __init__(
        self, original_label: bool, label_after_attack: bool, semantic_similarity: float
    ):
        self.original_label = original_label
        self.label_after_attack = label_after_attack
        self.semantic_similarity = semantic_similarity

    def successfully_attacked(self) -> bool:
        return self.original_label != self.label_after_attack


class AttackAggregateMetrics:
    def __init__(
        self, success_rate: float, avg_semantic_similarity: float, n_skipped: int
    ):
        self.success_rate = success_rate
        self.avg_semantic_similarity = avg_semantic_similarity

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
        print(
            f"Success rate: {self.success_rate}, average semantic similarity between"
            f" a sentence and its adversarial example: {self.avg_semantic_similarity}"
        )
