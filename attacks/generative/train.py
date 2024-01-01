import copy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.generative.model import GenerativeAttacker
from models.base import FakeNewsDetector
from src.evaluate_attack import DATASETS_DICT, MODELS_DICT
from src.metrics import SimilarityEvaluator
from src.torch_utils import get_available_torch_device


class PPOTrainer:
    def __init__(self, model: GenerativeAttacker):
        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()

    def set_reference_model(self, model: GenerativeAttacker):
        self.reference_model = copy.deepcopy(model)


def all_equal(values) -> bool:
    return len(values) == 0 or all([len(value) == len(values[0]) for value in values])


def train_iteration(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    dataloader: DataLoader,
    lr: float,
    device: str,
    common_max_length: int,
    similarity_evaluator: SimilarityEvaluator,
) -> None:
    attacker.train()
    victim.eval()
    ppo_trainer = PPOTrainer(attacker)
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc="train iteration",
        leave=False,
    ):
        input_ids = batch["attacker_prompt_ids"].to(device)
        generated_ids, token_logits, reference_logits = attacker.generate(
            input_ids, common_max_length, ppo_trainer.reference_model.bert
        )
        batch_prefixes = attacker.decode_prefixes(generated_ids)
        victim_classification_probabilities: list[torch.Tensor] = []
        for i in range(len(batch_prefixes)):
            assert all_equal(
                [len(token_logits[i]), len(reference_logits[i]), len(batch_prefixes[i])]
            )
        with torch.no_grad():
            for sample_prefixes in batch_prefixes:
                sample_probabilities: list[torch.Tensor] = []
                for prefix in sample_prefixes:
                    sample_probabilities.append(victim.get_probabilities(prefix))
                victim_classification_probabilities.append(torch.stack(sample_probabilities))
            target_labels = 1 - batch["label"]
            target_label_probabilities = [
                torch.stack(
                    [
                        victim_classification_probabilities[i][j][target_labels[i]]
                        for j in range(len(victim_classification_probabilities[i]))
                    ]
                )
                for i in range(len(target_labels))
            ]

            original_seqs = attacker.tokenizer.batch_decode(
                batch["attacker_prompt_ids"], skip_special_tokens=True
            )
            similarity_scores = [
                torch.Tensor(
                    [
                        similarity_evaluator.evaluate(prefix, original_seq)
                        for prefix in sample_prefixes
                    ]
                )
                for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
            ]

        # todo maybe include naturalness as well but tbd
        # todo make sure those tensors have equal length
        rewards = [
            target_label_probabilities[i] + similarity_scores[i] for i in range(len(batch_prefixes))
        ]
        pass

        # calculate the loss and implement the weights update with rewards,
        # token logits, and reference logits


def eval_iteration(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    dataloader: DataLoader,
    save_path: Path,
    device: str,
    common_max_length: int,
    similarity_evaluator: SimilarityEvaluator,
) -> None:
    attacker.eval()
    # monitor all loss components separately (naturality, similarity etc.)
    pass
    # get an article from the dataset
    # generate response to it (feed with label), retain logits
    # measure eval loss, avg. similarity, fooling factor, naturalness


def train(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    lr: float,
    save_path: Path,
    device: str,
    common_max_length: int,
    similarity_evaluator: SimilarityEvaluator,
) -> None:
    victim.eval()
    for i in tqdm(range(n_epochs), desc="training..."):
        train_iteration(
            attacker, victim, train_dataloader, lr, device, common_max_length, similarity_evaluator
        )
        eval_iteration(
            attacker,
            victim,
            eval_dataloader,
            save_path,
            device,
            common_max_length,
            similarity_evaluator,
        )


def main(
    victim_class: str,
    victim_config: dict[str, Any],
    victim_weights_path: Path,
    attacker_config: dict[str, Any],
    train_split_path: Path,
    eval_split_path: Path,
    n_epochs: int,
    lr: float,
    batch_size: int,
    save_path: Path,
    similarity_evaluator_name: str,
) -> None:
    device = get_available_torch_device()
    victim_config["device"] = device
    victim = MODELS_DICT[victim_class](**victim_config)
    victim.load_state_dict(torch.load(victim_weights_path, map_location=torch.device(device)))
    victim.to(device)
    victim.eval()
    attacker = GenerativeAttacker.from_config(attacker_config)
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, device)

    common_max_length = min(attacker.max_length, victim.max_length)

    train_dataset = DATASETS_DICT[victim_class](
        train_split_path,
        attacker.tokenizer,
        common_max_length,
        include_logits=True,
        attacker_tokenizer=attacker.tokenizer,
    )
    eval_dataset = DATASETS_DICT[victim_class](
        eval_split_path,
        attacker.tokenizer,
        common_max_length,
        include_logits=True,
        attacker_tokenizer=attacker.tokenizer,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    train(
        attacker,
        victim,
        train_dataloader,
        eval_dataloader,
        n_epochs,
        lr,
        save_path,
        device,
        common_max_length,
        similarity_evaluator,
    )


if __name__ == "__main__":
    generative_params = yaml.safe_load(open("params.yaml"))["attacks.generative"]
    victim_class = generative_params["victim"]
    victim_config = yaml.safe_load(open("configs/model_configs.yaml"))[victim_class]
    victim_weights_path = Path(generative_params["victim_weights_path"])
    attacker_config = yaml.safe_load(open("configs/attacker_configs.yaml"))["generative"]
    train_split_path = Path(generative_params["train_split_path"])
    eval_split_path = Path(generative_params["eval_split_path"])
    n_epochs = int(generative_params["n_epochs"])
    lr = float(generative_params["lr"])
    batch_size = int(generative_params["batch_size"])
    save_path = Path(generative_params["save_path"])
    similarity_evaluator_name = generative_params["similarity_evaluator_name"]
    main(
        victim_class,
        victim_config,
        victim_weights_path,
        attacker_config,
        train_split_path,
        eval_split_path,
        n_epochs,
        lr,
        batch_size,
        save_path,
        similarity_evaluator_name,
    )
