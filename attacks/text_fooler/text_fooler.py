import json
from pathlib import Path

import numpy as np
import stanza
import torch
from nltk.tokenize import word_tokenize

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
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name)
        self.neighbors_path = neighbors_path
        self.n_neighbors_considered = n_neighbors_considered
        self.pos_tagger = stanza.Pipeline(
            "en",
            processors="tokenize,mwt,pos",
            use_gpu=True,
            tokenize_pretokenized=True,
        )
        if not neighbors_path.exists():
            precompute_neighbors(n_neighbors_precomputed, neighbors_path)

    @classmethod
    def from_config(cls, config: dict) -> "AdversarialAttacker":
        return TextFoolerAttacker(
            similarity_threshold=float(config["similarity_threshold"]),
            similarity_evaluator_name=config["similarity_evaluator_name"],
            neighbors_path=Path(config["neighbors_path"]),
            n_neighbors_considered=int(config["n_neighbors_considered"]),
            n_neighbors_precomputed=int(config["n_neighbors_precomputed"]),
        )

    def get_importance_scores(self, words: list[str], model: FakeNewsDetector):
        original_logits = model.get_logits("".join(words))
        original_label = torch.argmax(original_logits).item()
        scores: list[float] = []
        for i in range(len(words)):
            remaining_words = words[:i] + words[i + 1:]
            new_logits = model.get_logits("".join(remaining_words))
            new_label = torch.argmax(new_logits).item()
            if new_label != original_label:
                score = original_logits[original_label] - new_logits[original_label]
            else:
                score = (
                        original_logits[original_label]
                        - new_logits[original_label]
                        + new_logits[new_label]
                        - original_logits[new_label]
                )
            scores.append(score)
        return scores

    def get_pos_tags(self, sentence: str):
        return [word.upos for word in self.pos_tagger(sentence).sentences[0].words]

    def get_confidence_and_similarity_scores(
            self,
            sentence_words: list[str],
            replaced_index: int,
            candidates: list[str],
            model: FakeNewsDetector,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_sentence = " ".join(sentence_words)
        original_logits = model.get_logits(original_sentence)
        original_label = torch.argmax(original_logits).item()
        original_pos = self.get_pos_tags(original_sentence)
        assert (len(original_pos)) == len(sentence_words)

        influence_scores: list[float] = []
        similarity_scores: list[float] = []

        for candidate in candidates:
            sentence_words[replaced_index] = candidate
            substituted_sentence = " ".join(sentence_words)
            substituted_logits = model.get_logits(substituted_sentence)

            influence_score = substituted_logits[original_label]

            substitute_pos = self.get_pos_tags(substituted_sentence)
            if substitute_pos != original_pos:
                similarity_score = 0.
            else:
                similarity_score = self.similarity_evaluator.evaluate(
                    original_sentence, substituted_sentence
                )
            influence_scores.append(influence_score)
            similarity_scores.append(similarity_score)

        return np.array(influence_scores), np.array(similarity_scores)

    def generate_adversarial_example(self, text: str, model: FakeNewsDetector) -> str:
        words = word_tokenize(text)
        importance_scores = np.array(self.get_importance_scores(words, model))

        with open(self.neighbors_path) as f:
            neighbors = json.load(f)

        # reverse sorting
        importance_indices = np.argsort(-1 * importance_scores)
        for i in importance_indices:
            if (
                    self.similarity_evaluator.evaluate(text, " ".join(words))
                    > self.similarity_threshold
            ):
                break

            if words[i] not in neighbors:
                continue

            candidates = words[i][: self.n_neighbors_considered]
            confidences, similarities = self.get_confidence_and_similarity_scores(
                words, i, candidates, model
            )

            high_similarity_candidate_indices = np.where(
                similarities > self.similarity_threshold
            )

            if not high_similarity_candidate_indices:
                break

            candidates = [candidates[i] for i in high_similarity_candidate_indices]
            confidences = confidences[high_similarity_candidate_indices]
            similarities = similarities[high_similarity_candidate_indices]

            successful_candidate_indices = np.where(confidences < 0.5)
            if successful_candidate_indices:
                highest_similarity_index = similarities.argmax()
                words[i] = candidates[highest_similarity_index]
                break
            else:
                lowest_confidence_index = confidences.argmin()
                words[i] = candidates[lowest_confidence_index]

        return " ".join(words)
