from typing import Type

import numpy as np
import torch
from nltk.tokenize import word_tokenize

from attacks.base import AdversarialAttacker
from models.base import FakeNewsDetector
from src.evaluate_attack import SimilarityEvaluator


class TextFoolerAttacker(AdversarialAttacker):
    def __init__(self, similarity_threshold: float, similarity_evaluator_name: str):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name)

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
                score = original_logits[original_label] - new_logits[original_label] + \
                        new_logits[new_label] - original_logits[new_label]
            scores.append(score)
        return scores

    def generate_adversarial_example(self, text: str, model: FakeNewsDetector) -> str:
        words = word_tokenize(text)
        importance_scores = np.array(self.get_importance_scores(words, model))

        # reverse sorting
        importance_indices = np.argsort(-1 * importance_scores)
        for i in importance_indices:
            if (
                    self.similarity_evaluator.evaluate(text, " ".join(words))
                    > self.similarity_threshold
            ):
                break
        # extract N candidates - words with the same POS tag, and high cosine
        #     similarity
        #     note: POS depends on the context so to filter that out,
        #       fist make the replacement and then check if the same pos
        #   for each candidate, get the influence on model prediction and
        #     similarity score with the sentence after replacement
        #   filter out candidates whose similarity is too low
        #   if there exist candidates that fool the model, take the one with
        #     highest similarity
        #   otherwise, take the one with the biggest influence on prediction
        #     note: this could instead be balanced to choose a word that
        #     both keeps the similarity high and affects the score a lot
        return " ".join(words)
