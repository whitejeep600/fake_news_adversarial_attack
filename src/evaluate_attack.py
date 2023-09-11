from pathlib import Path
from typing import Type

import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from attacks.base import AdversarialAttacker
from attacks.trivial import TrivialAttacker
from models.base import FakeNewsDetector
from models.baseline.dataset import FakeNewsDataset
from models.baseline.model import BaselineBert
from src.metrics import AttackAggregateMetrics, AttackSingleSampleMetrics
from src.torch_utils import get_available_torch_device

ATTACKERS_DICT: dict[str, Type[AdversarialAttacker]] = {"trivial": TrivialAttacker}

MODELS_DICT: dict[str, Type[FakeNewsDetector]] = {"baseline": BaselineBert}


class SimilarityEvaluator:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, text1: str, text2: str) -> float:
        embeddings = [
            self.model.encode(text, convert_to_tensor=True) for text in [text1, text2]
        ]
        return torch.dot(embeddings[0], embeddings[1]).item()


# todo code structure probably different for easy experiments but let's develop
#  say two models and two attackers first and then see how to organize it neatly


# todo maybe move some of the dataset management to the model, in particular make
#  sure the same data is loaded from the source .csv (no author and so on)
def main(
    eval_split_path: Path,
    model_name: str,
    model_class: str,
    weights_path: Path,
    attacker_name: str,
    similarity_evaluator_name: str,
    max_length: int,
):
    if attacker_name not in ATTACKERS_DICT.keys():
        raise ValueError("Unsupported attacker name")
    if model_class not in MODELS_DICT.keys():
        raise ValueError("Unsupported model name")
    device = get_available_torch_device()
    attacker = ATTACKERS_DICT[attacker_name]()
    model = MODELS_DICT[model_class](model_name, 2)
    model.load_state_dict(
        torch.load(
            weights_path, map_location=torch.device(device)
        )
    )
    model.to(device)
    model.eval()
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name)

    # todo try to kick that out to the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_dataset = FakeNewsDataset(eval_split_path, tokenizer, max_length)
    n_skipped = 0
    metrics: list[AttackSingleSampleMetrics] = []
    for sample in tqdm(
        eval_dataset.iterate_untokenized(),
        total=len(eval_dataset),
        desc="Running adversarial attack evaluation",
    ):
        original_text = sample["text"]
        label = sample["label"]
        tokenized_text = tokenizer(
            original_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        model_prediction = model.get_label(
            tokenized_text["input_ids"].to(device),
            tokenized_text["attention_mask"].to(device),
        )
        if label != model_prediction:
            n_skipped += 1
            continue
        adversarial_example = attacker.generate_adversarial_example(
            original_text, model
        )
        tokenized_adversarial_text = tokenizer(
            adversarial_example,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        adversarial_prediction = model.get_label(
            tokenized_adversarial_text["input_ids"].to(device),
            tokenized_adversarial_text["attention_mask"].to(device),
        )
        semantic_similarity = similarity_evaluator.evaluate(
            original_text, adversarial_example
        )
        metrics.append(
            AttackSingleSampleMetrics(
                label, adversarial_prediction, semantic_similarity
            )
        )

    aggregate_metrics = AttackAggregateMetrics.from_aggregation(metrics)
    aggregate_metrics.print_summary()


if __name__ == "__main__":
    evaluation_params = yaml.safe_load(open("params.yaml"))["src.evaluate_attack"]
    eval_split_path = Path(evaluation_params["eval_split_path"])
    model_name = evaluation_params["model_name"]
    model_class = evaluation_params["model_class"]
    weights_path = Path(evaluation_params["weights_path"])
    attacker_name = evaluation_params["attacker_name"]
    similarity_evaluator_name = evaluation_params["similarity_evaluator_name"]
    max_length = int(evaluation_params["max_length"])
    main(
        eval_split_path,
        model_name,
        model_class,
        weights_path,
        attacker_name,
        similarity_evaluator_name,
        max_length,
    )
