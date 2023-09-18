import json
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torch import Tensor

from attacks.base import AdversarialAttacker
from attacks.text_fooler.precompute_neighbors import main as precompute_neighbors
from models.base import FakeNewsDetector
from src.metrics import SimilarityEvaluator


class TextFoolerAttacker(AdversarialAttacker):
    def __init__(
        self,
        similarity_threshold: float,
        similarity_evaluator_name: str,
        neighbors_path: Path,
        n_neighbors_considered: int,
        n_neighbors_precomputed: int,
        device: str,
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.similarity_evaluator = SimilarityEvaluator(
            similarity_evaluator_name, device
        )
        self.neighbors_path = neighbors_path
        self.n_neighbors_considered = n_neighbors_considered
        if not neighbors_path.exists():
            print(
                f"Precomputed word neighbors not found. They will be computed now"
                f" and cached under {neighbors_path}"
            )
            precompute_neighbors(n_neighbors_precomputed, neighbors_path)

    @classmethod
    def from_config(cls, config: dict) -> "AdversarialAttacker":
        return TextFoolerAttacker(
            similarity_threshold=float(config["similarity_threshold"]),
            similarity_evaluator_name=config["similarity_evaluator_name"],
            neighbors_path=Path(config["neighbors_path"]),
            n_neighbors_considered=int(config["n_neighbors_considered"]),
            n_neighbors_precomputed=int(config["n_neighbors_precomputed"]),
            device=config["device"],
        )

    def get_importance_scores(self, words: list[str], model: FakeNewsDetector):
        original_logits = model.get_logits("".join(words))
        original_label = int(torch.argmax(original_logits).item())
        scores: list[float] = []
        for i in range(len(words)):
            remaining_words = words[:i] + words[i + 1 :]
            new_logits = model.get_logits("".join(remaining_words))
            new_label = int(torch.argmax(new_logits).item())
            if new_label != original_label:
                score = original_logits[original_label] - new_logits[original_label]
            else:
                score = (
                    original_logits[original_label]
                    - new_logits[original_label]
                    + new_logits[new_label]
                    - original_logits[new_label]
                )
            scores.append(float(score))
        return scores

    def get_confidence_scores(
        self,
        sentence_words: list[str],
        replaced_index: int,
        candidates: list[str],
        model: FakeNewsDetector,
        original_logits: Tensor,
    ) -> np.ndarray:
        original_label = int(torch.argmax(original_logits).item())
        confidence_scores: list[float] = []
        for candidate in candidates:
            sentence_words[replaced_index] = candidate
            substituted_sentence = " ".join(sentence_words)
            substituted_logits = model.get_logits(substituted_sentence)

            confidence_score = substituted_logits[original_label].item()

            confidence_scores.append(confidence_score)

        return np.array(confidence_scores)

    def get_similarity_scores(
        self,
        sentence_words: list[str],
        replaced_index: int,
        candidates: list[str],
    ) -> np.ndarray:
        similarity_scores: list[float] = []

        for candidate in candidates:
            sentence_words[replaced_index] = candidate
            substituted_sentence = " ".join(sentence_words)

            similarity_score = self.similarity_evaluator.compare_to_reference(
                substituted_sentence
            )
            similarity_scores.append(similarity_score)

        return np.array(similarity_scores)

    def generate_adversarial_example(self, text: str, model: FakeNewsDetector) -> str:
        words = word_tokenize(text)
        importance_scores = np.array(self.get_importance_scores(words, model))

        self.similarity_evaluator.set_reference_sentence(text)

        with open(self.neighbors_path) as f:
            neighbors = json.load(f)

        # reverse sorting
        importance_indices = np.argsort(-1 * importance_scores)
        for i in importance_indices:
            if (
                self.similarity_evaluator.compare_to_reference(" ".join(words))
                < self.similarity_threshold
            ):
                break

            if words[i] not in neighbors:
                continue

            candidates = neighbors[words[i]][: self.n_neighbors_considered]
            original_logits = model.get_logits(text)

            similarities = self.get_similarity_scores(words, i, candidates)

            high_similarity_candidate_indices = np.where(
                similarities > self.similarity_threshold
            )[0]

            if len(high_similarity_candidate_indices) == 0:
                continue

            candidates = [candidates[i] for i in high_similarity_candidate_indices]
            similarities = similarities[high_similarity_candidate_indices]

            confidences = self.get_confidence_scores(
                words, i, candidates, model, original_logits
            )

            successful_candidate_indices = np.where(confidences < 0.5)[0]
            if successful_candidate_indices:
                highest_similarity_index = similarities.argmax()
                words[i] = candidates[highest_similarity_index]
                break
            else:
                lowest_confidence_index = confidences.argmin()
                words[i] = candidates[lowest_confidence_index]

        self.similarity_evaluator.reset_reference_sentence()
        return " ".join(words)
