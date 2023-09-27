import json
import re
from copy import copy
from pathlib import Path

import numpy as np
import shap
import torch

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
        self.similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, device)
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
        original_probabilities = model.get_probabilities(" ".join(words))
        original_label = int(torch.argmax(original_probabilities).item())
        scores: list[float] = []
        for i in range(len(words)):
            remaining_words = words[:i] + words[i + 1 :]
            new_probabilities = model.get_probabilities(" ".join(remaining_words))
            new_label = int(torch.argmax(new_probabilities).item())
            if new_label == original_label:
                score = original_probabilities[original_label] - new_probabilities[original_label]
            else:
                score = (
                    original_probabilities[original_label]
                    - new_probabilities[original_label]
                    + new_probabilities[new_label]
                    - original_probabilities[new_label]
                )
            scores.append(float(score))
        return scores

    def get_confidence_scores(
        self,
        sentence_tokens: list[str],
        replaced_index: int,
        candidates: list[str],
        model: FakeNewsDetector,
        original_label: int,
    ) -> np.ndarray:
        sentence_tokens = copy(sentence_tokens)
        confidence_scores: list[float] = []
        for candidate in candidates:
            sentence_tokens[replaced_index] = candidate
            substituted_sentence = "".join(sentence_tokens)
            substituted_probabilities = model.get_probabilities(substituted_sentence)
            confidence_score = substituted_probabilities[original_label].item()
            confidence_scores.append(confidence_score)

        return np.array(confidence_scores)

    def get_similarity_scores(
        self,
        sentence_tokens: list[str],
        replaced_index: int,
        candidates: list[str],
    ) -> np.ndarray:
        sentence_tokens = copy(sentence_tokens)
        similarity_scores: list[float] = []

        for candidate in candidates:
            sentence_tokens[replaced_index] = candidate
            substituted_sentence = "".join(sentence_tokens)

            similarity_score = self.similarity_evaluator.compare_to_reference(substituted_sentence)
            similarity_scores.append(similarity_score)

        return np.array(similarity_scores)

    def get_neighbors(self, token: str, all_neighbors: dict[str, list[str]]) -> list[str] | None:
        if len(token) == 0:
            return None
        whitespace_split = re.split(r"(\S+)", token)
        leading_whitespace = whitespace_split[0]
        trailing_whitespace = whitespace_split[-1]
        is_uppercase = token.isupper()
        is_capitalized = token[0].isupper()
        base_token = token.strip().lower()
        if base_token not in all_neighbors:
            return None
        candidates = all_neighbors[base_token][: self.n_neighbors_considered]
        candidates = [
            leading_whitespace + candidate + trailing_whitespace for candidate in candidates
        ]
        if is_uppercase:
            return [candidate.upper() for candidate in candidates]
        elif is_capitalized:
            return [candidate[0].upper() + candidate[1:] for candidate in candidates]
        else:
            return candidates

    def generate_adversarial_example(self, text: str, model: FakeNewsDetector) -> str:
        original_label = model.get_prediction(text)
        pipeline = model.to_pipeline()
        explainer = shap.Explainer(pipeline)
        shap_values = explainer([text])
        tokens = shap_values.data[0]
        shap_values = shap_values.values[0]

        # We want to attack the words that move the model's the most in the direction
        # of the original label.
        importance_scores = shap_values[:, original_label]

        self.similarity_evaluator.set_reference_sentence(text)

        with open(self.neighbors_path) as f:
            all_neighbors = json.load(f)

        # reverse sorting
        importance_indices = np.argsort(-1 * importance_scores)
        for i in importance_indices:
            if (
                self.similarity_evaluator.compare_to_reference("".join(tokens))
                < self.similarity_threshold
            ):
                break

            candidates = self.get_neighbors(tokens[i], all_neighbors)
            if candidates is None:
                continue

            similarities = self.get_similarity_scores(tokens, i, candidates)

            high_similarity_candidate_indices = np.where(similarities > self.similarity_threshold)[
                0
            ]

            if len(high_similarity_candidate_indices) == 0:
                continue

            candidates = [candidates[i] for i in high_similarity_candidate_indices]
            similarities = similarities[high_similarity_candidate_indices]

            confidences = self.get_confidence_scores(tokens, i, candidates, model, original_label)

            successful_candidate_indices = np.where(confidences < 0.5)[0]
            if len(successful_candidate_indices):
                similarities = similarities[successful_candidate_indices]
                candidates = [candidates[i] for i in successful_candidate_indices]
                highest_similarity_index = similarities.argmax()
                tokens[i] = candidates[highest_similarity_index]
                break
            else:
                lowest_confidence_index = confidences.argmin()
                best_candidate = candidates[lowest_confidence_index]
                # frag = "".join(tokens[i - 3 : i + 3])
                # print(f"Replacing {tokens[i]} with {best_candidate}, fragment {frag}\n")
                # confidence = confidences[lowest_confidence_index]
                # similarity = similarities[lowest_confidence_index]
                # print(f"Confidence now {confidence}, similarity {similarity}")
                tokens[i] = best_candidate

        self.similarity_evaluator.reset_reference_sentence()
        return "".join(tokens)
