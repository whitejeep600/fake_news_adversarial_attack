import copy
import json
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from matplotlib import pyplot as plt
from numpy import mean
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

# Policy update clipping threshold
EPSILON = 0.2

# 2000 years and we're still using Greek if we want something to sound smart.

TRAIN = "train"
EVAL = "eval"

MODES = [TRAIN, EVAL]


REWARD_METRIC = "reward"
POLICY_LOSS_METRIC = "policy_loss"
VALUE_LOSS_METRIC = "value_loss"


def all_equal(values) -> bool:
    return len(values) == 0 or all([value == values[0] for value in values])


# Just a util to automatically create a target list if it doesn't exist, yo
class ListDict:
    def __init__(self):
        self.lists: dict[str, list] = {}

    def append(self, list_name: str, item):
        if list_name not in self.lists.keys():
            self.lists[list_name] = []
        self.lists[list_name].append(item)

    def __getitem__(self, item):
        return self.lists[item]


class PPOTrainer:
    def __init__(
        self,
        trained_model: GenerativeAttacker,
        rewards_and_metrics_function: Callable,
        value_model: ValueModel,
        attacker_optimizer: Optimizer,
        value_optimizer: Optimizer,
        device: str,
        stats_save_dir: Path
    ):
        self.trained_model = trained_model
        self.rewards_and_metrics_function = rewards_and_metrics_function
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()
        self.trained_model.bert.to(device)
        self.reference_model.bert.to(device)
        self.value_model = value_model
        self.attacker_optimizer = attacker_optimizer
        self.value_optimizer = value_optimizer
        self.device = device
        self.stats_save_dir = stats_save_dir
        self.standard_metric_names = [
            REWARD_METRIC,
            POLICY_LOSS_METRIC,
            VALUE_LOSS_METRIC,
        ]
        self.all_data: dict[str, dict[str, list[float]]] = {
            metric_name: {
                TRAIN: [],
                EVAL: []
            }
            for metric_name in self.standard_metric_names
        }
        self.train_start_time = time.time()
        self.epochs_elapsed = 0

    def freeze_reference_model(self):
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()

    def decode_tokens_and_get_logits(
        self, batch: torch.Tensor, max_length: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        self.attacker_optimizer.zero_grad()
        batch = batch.to(self.device)
        torch.set_grad_enabled(True)
        generated_ids: list[torch.Tensor] = []
        token_probabilities: list[torch.Tensor] = []
        reference_probabilities: list = []
        for seq in batch:
            new_ids, scores = self.trained_model.generate_with_greedy_decoding(
                seq.unsqueeze(0), max_length
            )
            generated_ids.append(new_ids)
            token_probabilities.append(
                torch.stack(
                    [
                        torch.softmax(scores[i][0], dim=0)[new_ids[i + 1]]
                        for i in range(len(scores))
                    ],
                )
            )
            reference_probabilities.append(
                torch.exp(
                    self.reference_model.bert.compute_transition_scores(
                        new_ids.unsqueeze(0), scores, normalize_logits=True
                    ).squeeze(0)
                )
            )
        return generated_ids, token_probabilities, reference_probabilities

    def decode_prefixes(self, generated_ids: list[torch.Tensor]) -> list[list[str]]:
        return self.trained_model.decode_prefixes(generated_ids)

    def get_value_function_scores(
        self, batch_prefixes: list[list[str]], original_seqs: list[str]
    ) -> list[torch.Tensor]:
        self.value_optimizer.zero_grad()
        return [
            torch.concat(
                [self.value_model.get_value(prefix, original_seq) for prefix in sample_prefixes]
            )
            for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
        ]

    def get_clipped_objectives(
        self,
        rewards: list[torch.Tensor],
        values: list[torch.Tensor],
        token_probs: list[torch.Tensor],
        reference_probs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        max_generated_length = max([len(reward_tensor) for reward_tensor in rewards])
        with torch.no_grad():
            discount_exponents = torch.pow(GAMMA * LAMBDA, torch.arange(max_generated_length)).to(self.device)
            # Following the notation and equations from Schulman et al.
            batch_size = len(rewards)
            gammas = [
                rewards[i][:-1] + GAMMA * values[i][1:] - values[i][:-1] for i in range(batch_size)
            ]
            advantages = [
                torch.stack(
                    [
                        torch.sum(
                            gammas[batch_index][t:] * discount_exponents[: len(gammas[batch_index][t:])]
                        )
                        for t in range(len(rewards[batch_index]))
                    ],
                    dim=0
                )
                for batch_index in range(batch_size)
            ]
        ratios = [token_probs[i] / reference_probs[i] for i in range(batch_size)]
        clipped_ratios = [torch.clip(ratio, 1 - EPSILON, 1 + EPSILON) for ratio in ratios]
        clipped_objectives = [
            torch.minimum(ratios[i] * advantages[i], clipped_ratios[i] * advantages[i])
            for i in range(batch_size)
        ]
        return clipped_objectives

    def get_policy_loss(self, clipped_objectives: list[torch.Tensor]) -> torch.Tensor:
        # gradient ascent
        policy_loss = -1 * torch.mean(torch.concat(clipped_objectives))
        return policy_loss

    def policy_loss_step(self, policy_loss: torch.Tensor) -> None:
        policy_loss.backward()
        self.attacker_optimizer.step()

    def get_value_loss(
        self, rewards: list[torch.Tensor], values: list[torch.Tensor]
    ) -> torch.Tensor:
        value_loss = torch.mean(
            torch.concat([values[i] - rewards[i][-1].detach() for i in range(batch_size)])
        )
        return value_loss

    def value_loss_step(self, value_loss: torch.Tensor) -> None:
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

    def train(self) -> None:
        self.trained_model.train()
        self.value_model.train()
        self.reference_model.eval()

    def eval(self) -> None:
        self.trained_model.eval()
        self.value_model.eval()
        self.reference_model.eval()

    def initialize_iteration(self, mode: str):
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        if mode == TRAIN:
            self.train()
        else:
            self.eval()
        self.epochs_elapsed += 1

    def save_trained_model(self, path: Path) -> None:
        torch.save(self.trained_model.bert.state_dict(), path)

    def iteration(
            self,
            dataloader: DataLoader,
            device: str,
            common_max_length: int,
            mode: str,
    ) -> float:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"

        self.initialize_iteration(mode)
        if mode == TRAIN:
            self.freeze_reference_model()

        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_rewards: list[float] = []
        nonstandard_metrics = ListDict()

        for batch in tqdm(
                dataloader, total=len(dataloader), desc=f"{mode} iteration", leave=False, position=1
        ):
            input_ids = batch["attacker_prompt_ids"].to(device)
            generated_ids, token_probs, reference_probs = self.decode_tokens_and_get_logits(
                input_ids, common_max_length
            )
            batch_prefixes = self.decode_prefixes(generated_ids)

            for i in range(len(batch_prefixes)):
                assert all_equal(
                    [len(token_probs[i]), len(reference_probs[i]), len(batch_prefixes[i])])

            original_seqs = batch["attacker_prompt"]

            with torch.no_grad():
                rewards, stats = self.rewards_and_metrics_function(batch, batch_prefixes, original_seqs)

            for stat_name in stats.keys():
                nonstandard_metrics.append(stat_name, stats[stat_name])

            values = self.get_value_function_scores(batch_prefixes, original_seqs)

            clipped_objectives = self.get_clipped_objectives(
                rewards, values, token_probs, reference_probs
            )

            policy_loss = self.get_policy_loss(clipped_objectives)
            value_loss = self.get_value_loss(rewards, values)

            if mode == TRAIN:
                self.policy_loss_step(policy_loss)
                self.value_loss_step(value_loss)

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

            final_rewards = [reward[-1].item() for reward in rewards]

            epoch_rewards.append(float(mean(final_rewards)))

        mean_final_reward = float(mean(epoch_rewards))
        self.add_epoch_metrics(
            {
                REWARD_METRIC: mean_final_reward,
                POLICY_LOSS_METRIC: float(mean(epoch_policy_losses)),
                VALUE_LOSS_METRIC: float(mean(epoch_value_losses)),
            },
            mode
        )
        self.add_nonstandard_epoch_metrics(
            {
                list_name: float(mean(nonstandard_metrics[list_name])) for list_name in nonstandard_metrics.lists.keys()
            },
            mode
        )
        return mean_final_reward

    def add_epoch_metrics(
        self,
        epoch_metrics: dict[str, float],
        mode: str,
    ) -> None:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        for metric_name in epoch_metrics.keys():
            if metric_name in self.standard_metric_names:
                self.all_data[metric_name][mode].append(epoch_metrics[metric_name])
            else:
                warnings.warn(f"Metric '{metric_name}' is not standard, ignoring.")
        for metric_name in self.standard_metric_names:
            if metric_name not in epoch_metrics:
                warnings.warn(f"Metric '{metric_name}' was not passed for this iteration!")
        print(
            f"Epoch {self.epochs_elapsed},"
            f" this epoch's {mode} stats, as follows:\n"
            f"{epoch_metrics}\n"
        )
        if mode == EVAL:
            self.epochs_elapsed += 1

    def add_nonstandard_epoch_metrics(self, epoch_metrics: dict[str, float], mode: str) -> None:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        for metric_name in epoch_metrics.keys():
            if metric_name not in self.all_data:
                self.all_data[metric_name] = {
                    TRAIN: [],
                    EVAL: []
                }
            self.all_data[metric_name][mode].append(epoch_metrics[metric_name])

    def save_plots(self) -> None:
        plots_path = self.stats_save_dir / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        for variable in self.all_data.keys():
            for mode in MODES:
                title = f"{mode}_{variable}"
                plt.title(title)
                plt.plot(self.all_data[variable][mode])
                plt.xlabel("iteration")
                plt.savefig(plots_path / f"{title}.jpg")

    def save_logs(self) -> None:
        logs_path = self.stats_save_dir / "log.txt"
        with open(logs_path, "w") as f:
            f.write(json.dumps(self.all_data, indent=4))

    def save_summary(self, best_epoch_no: int) -> None:
        time_now = time.time()
        time_elapsed = time.gmtime(time_now - self.train_start_time)

        summary_path = self.stats_save_dir / "summary.txt"
        best_epoch_stats = {
            key: self.all_data[key][EVAL][best_epoch_no] for key in self.all_data.keys()
        }
        summary = (
            f"Training time: {time.strftime('%H:%M:%S', time_elapsed)}"
            f" Number of epochs elapsed: {self.epochs_elapsed}, best stats (final rewards)"
            f" for epoch {best_epoch_no}, as follows: {best_epoch_stats}"
        )
        with open(summary_path, "w") as f:
            f.write(summary)


def get_similarity_scores(
    batch_prefixes: list[list[str]],
    original_seqs: list[str],
    similarity_evaluator: SimilarityEvaluator,
    device: str
) -> list[torch.Tensor]:
    similarity_scores = [
        torch.Tensor(
            similarity_evaluator.evaluate_prefixes(sample_prefixes, original_seq)
        ).to(device)
        for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
    ]
    return similarity_scores


def get_fooling_factors(
    batch_prefixes: list[list[str]], target_labels: torch.Tensor, victim: FakeNewsDetector
) -> list[torch.Tensor]:
    victim_classification_probabilities: list[torch.Tensor] = []
    for sample_prefixes in batch_prefixes:
        sample_probabilities: list[torch.Tensor] = []
        for prefix in sample_prefixes:
            sample_probabilities.append(victim.get_probabilities(prefix))
        victim_classification_probabilities.append(torch.stack(sample_probabilities))
    # i.e. target label probabilities
    fooling_factors = [
        torch.stack(
            [
                victim_classification_probabilities[i][j][target_labels[i]]
                for j in range(len(victim_classification_probabilities[i]))
            ]
        )
        for i in range(len(batch_prefixes))
    ]
    return fooling_factors


def get_rewards_and_nonstandard_metrics(
        batch: dict,
        batch_prefixes: list[list[str]],
        original_seqs: list[str],
        similarity_evaluator: SimilarityEvaluator,
        victim: FakeNewsDetector,
        device: str
) -> tuple[list[torch.Tensor], dict[str, float]]:
    fooling_factors = get_fooling_factors(batch_prefixes, 1 - batch["label"], victim)
    similarity_scores = get_similarity_scores(batch_prefixes, original_seqs, similarity_evaluator, device)

    n_new_successful_attacks = sum(
        [sample_fooling_factors[-1].item() > 0.5 for sample_fooling_factors in
         fooling_factors]
    )
    success_rate = n_new_successful_attacks / len(original_seqs)
    rewards = [
        (fooling_factors[i] + similarity_scores[i]) / 2 for i in
        range(len(batch_prefixes))
    ]
    stats = {
        "fooling_factors": float(torch.mean(torch.concat(fooling_factors, dim=0)).item()),
        "similarity_scores": float(torch.mean(torch.concat(similarity_scores, dim=0)).item()),
        "success_rate": success_rate
    }
    return rewards, stats


def train(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    attacker_lr: float,
    value_lr: float,
    save_dir: Path,
    device: str,
    common_max_length: int,
    similarity_evaluator: SimilarityEvaluator,
    value_model: ValueModel,
) -> None:
    attacker_optimizer = AdamW(attacker.parameters(), lr=attacker_lr)
    value_optimizer = AdamW(value_model.parameters(), lr=value_lr)

    save_dir.mkdir(exist_ok=True, parents=True)
    model_save_path = save_dir / "ckpt.pt"

    victim.eval()
    similarity_evaluator.eval()
    rewards_and_metrics_function = partial(
        get_rewards_and_nonstandard_metrics,
        similarity_evaluator=similarity_evaluator,
        victim=victim,
        device=device
    )

    ppo_trainer = PPOTrainer(
        attacker,
        rewards_and_metrics_function,
        value_model,
        attacker_optimizer,
        value_optimizer,
        device,
        save_dir
    )
    best_mean_final_rewards: float | None = None
    best_epoch = -1

    for i in tqdm(range(n_epochs), desc="training...", position=0):
        ppo_trainer.iteration(train_dataloader, device, common_max_length, TRAIN)
        new_mean_final_rewards = ppo_trainer.iteration(
            eval_dataloader, device, common_max_length, EVAL
        )
        if best_mean_final_rewards is None or new_mean_final_rewards > best_mean_final_rewards:
            best_epoch = i
            best_mean_final_rewards = new_mean_final_rewards
            ppo_trainer.save_trained_model(model_save_path)

    ppo_trainer.save_logs()
    ppo_trainer.save_summary(best_epoch)
    ppo_trainer.save_plots()


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
    save_dir: Path,
    similarity_evaluator_name: str,
    value_model_name: str,
) -> None:
    device = get_available_torch_device()
    victim_config["device"] = device
    attacker_config["device"] = device
    victim = MODELS_DICT[victim_class](**victim_config)
    victim.load_state_dict(torch.load(victim_weights_path, map_location=torch.device(device)))
    victim.to(device)
    victim.eval()
    attacker = GenerativeAttacker.from_config(attacker_config)
    attacker.bert.to(device)
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, device)

    common_max_length = min(attacker.max_length, victim.max_length)

    value_model = ValueModel(value_model_name, common_max_length, device)

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
        save_dir,
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
    save_dir = Path(generative_params["save_dir"])
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
        save_dir,
        similarity_evaluator_name,
        value_model_name,
    )
