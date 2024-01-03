import copy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.generative.model import GenerativeAttacker
from attacks.generative.value_model import ValueModel
from models.base import FakeNewsDetector
from src.evaluate_attack import DATASETS_DICT, MODELS_DICT
from src.metrics import SimilarityEvaluator
from src.torch_utils import get_available_torch_device


# Discount factor, following the notation from the original PPO paper by Schulman et al.
GAMMA = 0.99

# Generalized Advantage Estimation factor
LAMBDA = 0.95

# Clipping threshold
EPSILON = 0.2

# 2000 years and we're still using Greek if we want something to sound smart.


# todo this is not really what this class does, do away with it or
#  write a new one that will be useful for different trainings
class PPOTrainer:
    def __init__(self, model: GenerativeAttacker):
        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()

    def set_reference_model(self, model: GenerativeAttacker):
        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()


def all_equal(values) -> bool:
    return len(values) == 0 or all([len(value) == len(values[0]) for value in values])


def train_iteration(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    dataloader: DataLoader,
    attacker_optimizer: Optimizer,
    value_optimizer: Optimizer,
    device: str,
    common_max_length: int,
    similarity_evaluator: SimilarityEvaluator,
    value_model: ValueModel,
) -> None:
    attacker.train()
    value_model.train()
    victim.eval()
    similarity_evaluator.eval()
    ppo_trainer = PPOTrainer(attacker)
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc="train iteration",
        leave=False,
    ):
        input_ids = batch["attacker_prompt_ids"].to(device)
        generated_ids, token_probs, reference_probs = attacker.generate(
            input_ids, common_max_length, ppo_trainer.reference_model.bert
        )
        batch_size = len(input_ids)
        batch_prefixes = attacker.decode_prefixes(generated_ids)
        victim_classification_probabilities: list[torch.Tensor] = []
        for i in range(len(batch_prefixes)):
            assert all_equal([len(token_probs[i]), len(reference_probs[i]), len(batch_prefixes[i])])
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
                for i in range(batch_size)
            ]

            original_seqs = attacker.tokenizer.batch_decode(
                batch["attacker_prompt_ids"], skip_special_tokens=True
            )
            # todo use set_reference_text to save  s o m e  time and memory at least
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
        rewards = [
            target_label_probabilities[i] + similarity_scores[i] for i in range(len(batch_prefixes))
        ]
        values = [
            torch.Tensor(
                [value_model.get_value(prefix, original_seq) for prefix in sample_prefixes]
            )
            for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
        ]
        pass
        max_generated_length = max([len(reward_tensor) for reward_tensor in rewards])
        discount_exponents = torch.pow(GAMMA * LAMBDA, torch.arange(max_generated_length))
        # Again following the notation and equations from Schulman et al.
        gammas = [
            rewards[i][:-1] + GAMMA * values[i][1:] - values[i][:-1] for i in range(batch_size)
        ]
        advantages = [
            torch.Tensor(
                [
                    torch.sum(
                        gammas[batch_index][t:] * discount_exponents[: len(gammas[batch_index][t:])]
                    )
                    for t in range(len(rewards[batch_index]))
                ]
            )
            for batch_index in range(batch_size)
        ]
        ratios = [token_probs[i] / reference_probs[i] for i in range(batch_size)]
        clipped_ratios = [torch.clip(ratio, 1 - EPSILON, 1 + EPSILON) for ratio in ratios]
        clipped_objectives = [
            torch.minimum(ratios[i] * advantages[i], clipped_ratios[i] * advantages[i])
            for i in range(batch_size)
        ]

        policy_loss = torch.mean(torch.concatenate(clipped_objectives))
        attacker_optimizer.zero_grad()
        policy_loss.backward()
        attacker_optimizer.step()

        value_loss = torch.mean(
            torch.concatenate([values[i] - rewards[i][-1] for i in range(batch_size)])
        )
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()


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
    # monitor all loss components separately (fooling, similarity etc.)
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
    attacker_lr: float,
    value_lr: float,
    save_path: Path,
    device: str,
    common_max_length: int,
    similarity_evaluator: SimilarityEvaluator,
    value_model: ValueModel,
) -> None:
    attacker_optimizer = AdamW(attacker.parameters(), lr=attacker_lr)
    value_optimizer = AdamW(value_model.parameters(), lr=value_lr)
    for i in tqdm(range(n_epochs), desc="training..."):
        train_iteration(
            attacker,
            victim,
            train_dataloader,
            attacker_optimizer,
            value_optimizer,
            device,
            common_max_length,
            similarity_evaluator,
            value_model,
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
    attacker_lr: float,
    value_lr: float,
    batch_size: int,
    save_path: Path,
    similarity_evaluator_name: str,
    value_model_name: str,
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

    value_model = ValueModel(value_model_name, common_max_length)

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
        attacker_lr,
        value_lr,
        save_path,
        device,
        common_max_length,
        similarity_evaluator,
        value_model,
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
    attacker_lr = float(generative_params["attacker_lr"])
    value_lr = float(generative_params["value_lr"])
    batch_size = int(generative_params["batch_size"])
    save_path = Path(generative_params["save_path"])
    similarity_evaluator_name = generative_params["similarity_evaluator_name"]
    value_model_name = generative_params["value_model_name"]
    main(
        victim_class,
        victim_config,
        victim_weights_path,
        attacker_config,
        train_split_path,
        eval_split_path,
        n_epochs,
        attacker_lr,
        value_lr,
        batch_size,
        save_path,
        similarity_evaluator_name,
        value_model_name,
    )
