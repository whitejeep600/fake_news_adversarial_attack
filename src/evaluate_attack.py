from pathlib import Path
from typing import Type

import torch
import yaml
from torch import multiprocessing
from tqdm import tqdm

from attacks.base import AdversarialAttacker
from attacks.text_fooler.text_fooler import TextFoolerAttacker
from models.base import FakeNewsDetector
from models.baseline.dataset import BaselineDataset
from models.baseline.model import BaselineBertDetector
from src.metrics import (
    AttackAggregateMetrics,
    AttackSingleSampleMetrics,
    SimilarityEvaluator,
    sample_metrics_to_dataframe,
)
from src.torch_utils import get_available_torch_device

ATTACKERS_DICT: dict[str, Type[AdversarialAttacker]] = {
    "text_fooler": TextFoolerAttacker,
}

MODELS_DICT: dict[str, Type[FakeNewsDetector]] = {"baseline": BaselineBertDetector}


# todo code structure probably different for easy experiments but let's develop
#  say two models and two attackers first and then see how to organize it neatly
# todo maybe move some of the dataset management to the model, in particular make
#  sure the same data is loaded from the source .csv (no author and so on). Or
#  create a super class FakeNewsDataset implementingm methods required for training
#  and for this evaluation script, and inherit from it in different models (e.g.
#  what is currently FakeNewsDataset will become BaselineDataset)


def process_sample(
    sample: dict,
    model: FakeNewsDetector,
    attacker: AdversarialAttacker,
    similarity_evaluator: SimilarityEvaluator,
) -> AttackSingleSampleMetrics | None:
    original_text = sample["text"]
    label = sample["label"]
    sample_id = sample["id"]
    model_prediction = model.get_prediction(original_text)
    if label != model_prediction:
        return None
    adversarial_example = attacker.generate_adversarial_example(original_text, model)
    adversarial_probabilities = model.get_probabilities(adversarial_example)
    adversarial_prediction = int(torch.argmax(adversarial_probabilities).item())
    semantic_similarity = similarity_evaluator.evaluate(original_text, adversarial_example)
    confidence = round(float(adversarial_probabilities[adversarial_prediction].item()), 2)
    return AttackSingleSampleMetrics(
        label,
        adversarial_prediction,
        round(semantic_similarity, 2),
        confidence,
        adversarial_example,
        sample_id,
    )


class AdversarialAttackProcess(multiprocessing.Process):
    def __init__(
        self,
        samples_q: multiprocessing.Queue,
        metrics_q: multiprocessing.Queue,
        model_class: str,
        model_config: dict,
        weights_path: Path,
        attacker_name: str,
        attacker_config: dict,
        similarity_evaluator_name: str,
        device: str,
    ):
        super().__init__()
        attacker_config["device"] = device
        model_config["device"] = device
        self.samples_q = samples_q
        self.metrics_q = metrics_q
        self.attacker = ATTACKERS_DICT[attacker_name].from_config(attacker_config)
        self.model = MODELS_DICT[model_class](**model_config)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        self.model.to(device)
        self.model.eval()
        self.similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, device)

    def run(self):
        torch.set_num_threads(1)
        self.model.eval()
        while True:
            sample = self.samples_q.get()
            if sample is None:
                break
            metrics = process_sample(sample, self.model, self.attacker, self.similarity_evaluator)
            self.metrics_q.put(metrics)


def main(
    eval_split_path: Path,
    model_class: str,
    model_config: dict,
    weights_path: Path,
    attacker_name: str,
    attacker_config: dict,
    similarity_evaluator_name: str,
    results_save_path: Path,
):
    multiprocessing.set_start_method("spawn")
    if attacker_name not in ATTACKERS_DICT.keys():
        raise ValueError("Unsupported attacker name")
    if model_class not in MODELS_DICT.keys():
        raise ValueError("Unsupported model name")

    model_config["device"] = get_available_torch_device()
    model = MODELS_DICT[model_class](**model_config)
    eval_dataset = BaselineDataset(eval_split_path, model.tokenizer, model.max_length)

    # None if sample skipped because the model's prediction was already wrong
    metrics: list[AttackSingleSampleMetrics | None] = []

    n_samples = len(eval_dataset)
    num_workers = multiprocessing.cpu_count()  # todo probably make that a parameter
    samples_q: multiprocessing.Queue = multiprocessing.Queue(maxsize=num_workers * 2)
    metrics_q: multiprocessing.Queue = multiprocessing.Queue(maxsize=n_samples)
    processes = [
        AdversarialAttackProcess(
            samples_q=samples_q,
            metrics_q=metrics_q,
            model_class=model_class,
            model_config=model_config,
            weights_path=weights_path,
            attacker_name=attacker_name,
            attacker_config=attacker_config,
            similarity_evaluator_name=similarity_evaluator_name,
            device=get_available_torch_device(),
        )
        for _ in range(num_workers)
    ]
    for proc in processes:
        proc.start()

    with tqdm(
        total=len(eval_dataset), desc="Running adversarial attack evaluation"
    ) as progress_bar:
        for sample in eval_dataset.iterate_untokenized():
            samples_q.put(sample)
            while not metrics_q.empty():
                next_metric = metrics_q.get()
                metrics.append(next_metric)
                progress_bar.update()

        for _ in range(num_workers):
            samples_q.put(None)

        while not len(metrics) == n_samples:
            next_metric = metrics_q.get()
            metrics.append(next_metric)
            progress_bar.update()

    for p in processes:
        p.join()

    non_skipped_metrics = [metric for metric in metrics if metric is not None]
    n_skipped = len(metrics) - len(non_skipped_metrics)
    aggregate_metrics = AttackAggregateMetrics.from_aggregation(non_skipped_metrics, n_skipped)
    aggregate_metrics.print_summary()
    result_df = sample_metrics_to_dataframe(non_skipped_metrics)
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
    attacker_config = yaml.safe_load(open("configs/attacker_configs.yaml"))[attacker_name]
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
        results_save_path,
    )
