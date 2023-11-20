from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.generative.model import GenerativeAttacker
from models.base import FakeNewsDetector
from src.evaluate_attack import MODELS_DICT, DATASETS_DICT
from src.torch_utils import get_available_torch_device


def train_iteration(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    dataloader: DataLoader,
    lr: float,
    device: str,
    common_max_length: int
) -> None:
    attacker.train()
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc="train iteration",
        leave=False,
    ):
        input_ids = batch["attacker_prompt_ids"].to(device)
        generated_ids = attacker.generate(input_ids, common_max_length)
        generated_seqs = attacker.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # debug
        original_seqs = [dataloader.dataset.df[dataloader.dataset.df["id"] == i]["text"].values for i in batch["id"].tolist()]
        victim_logits = [victim.get_logits(seq) for seq in generated_seqs]
        pass
        # get attacker logits
        # calculate the reward and loss
        # backpropagate


def eval_iteration(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    dataloader: DataLoader,
    save_path: Path,
    device: str,
    common_max_length: int
) -> None:
    attacker.eval()
    pass
    # get an article from the dataset
    # generate response to it (feed with label), retain logits
    # measure eval loss, avg. similarity, fooling factor, naturality


def train(
    attacker: GenerativeAttacker,
    victim: FakeNewsDetector,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    lr: float,
    save_path: Path,
    device: str,
    common_max_length: int
) -> None:
    victim.eval()
    for i in range(n_epochs):
        train_iteration(attacker, victim, train_dataloader, lr, device, common_max_length)
        eval_iteration(attacker, victim, eval_dataloader, save_path, device, common_max_length)


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
) -> None:
    device = get_available_torch_device()
    victim_config["device"] = device
    victim = MODELS_DICT[victim_class](**victim_config)
    victim.load_state_dict(torch.load(victim_weights_path, map_location=torch.device(device)))
    victim.to(device)
    victim.eval()
    attacker = GenerativeAttacker.from_config(attacker_config)

    common_max_length = min(attacker.max_length, victim.max_length)

    train_dataset = DATASETS_DICT[victim_class](
        train_split_path,
        attacker.tokenizer,
        common_max_length,
        include_logits=True,
        attacker_tokenizer=attacker.tokenizer,
    )
    eval_dataset = DATASETS_DICT[victim_class](
        train_split_path,
        attacker.tokenizer,
        common_max_length,
        include_logits=True,
        attacker_tokenizer=attacker.tokenizer,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    train(attacker, victim, train_dataloader, eval_dataloader, n_epochs, lr, save_path, device, common_max_length)


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
    )
