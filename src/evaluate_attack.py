from pathlib import Path
from typing import Type

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from attacks.base import AdversarialAttacker
from attacks.text_fooler.text_fooler import TextFoolerAttacker
from models.base import FakeNewsDetector
from models.baseline.dataset import FakeNewsDataset
from models.baseline.model import BaselineBert
from src.metrics import (
    AttackAggregateMetrics,
    AttackSingleSampleMetrics,
    SimilarityEvaluator,
)
from src.torch_utils import get_available_torch_device

ATTACKERS_DICT: dict[str, Type[AdversarialAttacker]] = {
    "text_fooler": TextFoolerAttacker,
}

MODELS_DICT: dict[str, Type[FakeNewsDetector]] = {"baseline": BaselineBert}


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
    results_save_path: Path
):
    if attacker_name not in ATTACKERS_DICT.keys():
        raise ValueError("Unsupported attacker name")
    if model_class not in MODELS_DICT.keys():
        raise ValueError("Unsupported model name")
    device = get_available_torch_device()
    attacker_config["device"] = device
    attacker = ATTACKERS_DICT[attacker_name].from_config(attacker_config)
    model_config["device"] = device
    model = MODELS_DICT[model_class](**model_config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, device)

    eval_dataset = FakeNewsDataset(eval_split_path, model.tokenizer, model.max_length)
    n_skipped = 0
    metrics: list[AttackSingleSampleMetrics] = []
    generated_examples: list[str] = []
    labels: list[int] = []
    confidences: list[float] = []
    similarities: list[float] = []
    sample_ids: list[int] = []
    succeeded: list[bool] = []

    for sample in tqdm(
        eval_dataset.iterate_untokenized(),
        total=len(eval_dataset),
        desc="Running adversarial attack evaluation",
    ):
        original_text = sample["text"]
        label = sample["label"]
        sample_id = sample["id"]
        model_prediction = model.get_prediction(original_text)
        if label != model_prediction:
            n_skipped += 1
            continue
        adversarial_example = attacker.generate_adversarial_example(
            original_text, model
        )
        adversarial_probabilities = model.get_probabilities(adversarial_example)
        adversarial_prediction = int(torch.argmax(adversarial_probabilities).item())
        semantic_similarity = similarity_evaluator.evaluate(
            original_text, adversarial_example
        )
        metrics.append(
            AttackSingleSampleMetrics(
                label, adversarial_prediction, semantic_similarity
            )
        )
        # print(original_text + "\n")
        # print(adversarial_example + "\n")
        # print(
        #     f"label {label}, adversarial prediction:"
        #     f" {adversarial_prediction}, similarity"
        #     f" {semantic_similarity}"
        # )
        generated_examples.append(adversarial_example)
        labels.append(adversarial_prediction)
        confidences.append(
            round(float(adversarial_probabilities[adversarial_prediction].item()), 2)
        )
        similarities.append(round(semantic_similarity, 2))
        sample_ids.append(sample_id)
        succeeded.append(label != adversarial_prediction)

    print(f"n skipped samples: {n_skipped}")
    aggregate_metrics = AttackAggregateMetrics.from_aggregation(metrics, n_skipped)
    aggregate_metrics.print_summary()
    result_df = pd.DataFrame(
        list(zip(sample_ids, generated_examples, labels, confidences, similarities, succeeded)),
        columns=["id", "text", "label", "confidence", "similarity", "succeeded"]
    )
    result_df.to_csv(results_save_path)


if __name__ == "__main__":
    evaluation_params = yaml.safe_load(open("params.yaml"))["src.evaluate_attack"]
    eval_split_path = Path(evaluation_params["eval_split_path"])
    model_class = evaluation_params["model_class"]
    model_config = yaml.safe_load(open("configs/model_configs.yaml"))[model_class]
    weights_path = Path(evaluation_params["weights_path"])
    attacker_name = evaluation_params["attacker_name"]
    results_save_path = Path(evaluation_params["results_save_path"])
    similarity_evaluator_name = evaluation_params["similarity_evaluator_name"]
    attacker_config = yaml.safe_load(open("configs/attacker_configs.yaml"))[
        attacker_name
    ]
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
        results_save_path
    )
