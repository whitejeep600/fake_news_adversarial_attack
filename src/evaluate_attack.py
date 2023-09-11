from pathlib import Path
from typing import Type

import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from attacks.base import AdversarialAttacker
from attacks.text_fooler import TextFoolerAttacker
from models.base import FakeNewsDetector
from models.baseline.dataset import FakeNewsDataset
from models.baseline.model import BaselineBert
from src.metrics import AttackAggregateMetrics, AttackSingleSampleMetrics
from src.torch_utils import get_available_torch_device

ATTACKERS_DICT: dict[str, Type[AdversarialAttacker]] = {
    "text_fooler": TextFoolerAttacker,
}

MODELS_DICT: dict[str, Type[FakeNewsDetector]] = {"baseline": BaselineBert}


class SimilarityEvaluator:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, text1: str, text2: str) -> float:
        embeddings = [
            self.model.encode(text, convert_to_tensor=True) for text in [text1, text2]
        ]
        return torch.cosine_similarity(embeddings[0], embeddings[1]).item()


# todo code structure probably different for easy experiments but let's develop
#  say two models and two attackers first and then see how to organize it neatly


# todo maybe move some of the dataset management to the model, in particular make
#  sure the same data is loaded from the source .csv (no author and so on)
def main(
    eval_split_path: Path,
    model_class: str,
    model_config: dict,
    weights_path: Path,
    attacker_name: str,
    attacker_config: dict,
    similarity_evaluator_name: str,
):
    if attacker_name not in ATTACKERS_DICT.keys():
        raise ValueError("Unsupported attacker name")
    if model_class not in MODELS_DICT.keys():
        raise ValueError("Unsupported model name")
    device = get_available_torch_device()
    attacker = ATTACKERS_DICT[attacker_name](**attacker_config)
    model_config["device"] = device
    model = MODELS_DICT[model_class](**model_config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name)

    eval_dataset = FakeNewsDataset(eval_split_path, model.tokenizer, model.max_length)
    n_skipped = 0
    metrics: list[AttackSingleSampleMetrics] = []
    for sample in tqdm(
        eval_dataset.iterate_untokenized(),
        total=len(eval_dataset),
        desc="Running adversarial attack evaluation",
    ):
        original_text = sample["text"]
        label = sample["label"]
        model_prediction = model.get_prediction(original_text)
        if label != model_prediction:
            n_skipped += 1
            continue
        adversarial_example = attacker.generate_adversarial_example(
            original_text, model
        )
        adversarial_prediction = model.get_prediction(adversarial_example)
        semantic_similarity = similarity_evaluator.evaluate(
            original_text, adversarial_example
        )
        metrics.append(
            AttackSingleSampleMetrics(
                label, adversarial_prediction, semantic_similarity
            )
        )

    aggregate_metrics = AttackAggregateMetrics.from_aggregation(metrics, n_skipped)
    aggregate_metrics.print_summary()


if __name__ == "__main__":
    evaluation_params = yaml.safe_load(open("params.yaml"))["src.evaluate_attack"]
    eval_split_path = Path(evaluation_params["eval_split_path"])
    model_class = evaluation_params["model_class"]
    model_config = yaml.safe_load(open("model_configs.yaml"))[model_class]
    weights_path = Path(evaluation_params["weights_path"])
    attacker_name = evaluation_params["attacker_name"]
    similarity_evaluator_name = evaluation_params["similarity_evaluator_name"]
    attacker_config = yaml.safe_load(open("attacker_configs.yaml"))[attacker_name]
    if attacker_config is None:
        attacker_config = {}
    main(
        eval_split_path,
        model_class,
        model_config,
        weights_path,
        attacker_name,
        attacker_config,
        similarity_evaluator_name,
    )
