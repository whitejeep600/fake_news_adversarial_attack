from pathlib import Path

import torch.nn
import yaml
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline.dataset import BaselineDataset
from models.baseline.model import BaselineBertDetector
from src.torch_utils import get_available_torch_device


def train_iteration(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    loss_fn: _Loss,
    optimizer: Optimizer,
    device: str,
) -> None:
    model.train()
    losses: list[float] = []
    for batch in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc="train iteration",
        leave=False,
    ):
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        prediction_logits = model(input_ids, attention_masks)
        loss = loss_fn(prediction_logits, labels)
        losses.append(loss)

        loss.backward()
        optimizer.step()

    avg_train_loss = sum(losses) / len(losses)
    print(f"\naverage train loss: {avg_train_loss}")


def calculate_stats(predictions: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int, int]:
    tp = int(torch.sum(torch.logical_and(predictions, labels)).item())
    fp = int(torch.sum(torch.logical_and(predictions, torch.logical_not(labels))).item())
    tn = int(
        torch.sum(
            torch.logical_and(
                torch.logical_not(predictions),
                torch.logical_not(labels),
            )
        ).item()
    )
    fn = int(torch.sum(torch.logical_and(torch.logical_not(predictions), labels)).item())
    return tp, fp, tn, fn


def eval_iteration(
    model: torch.nn.Module, eval_dataloader: DataLoader, loss_fn: _Loss, device: str
) -> float | None:
    model.eval()
    losses: list[float] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    with torch.no_grad():
        for batch in tqdm(
            eval_dataloader,
            total=len(eval_dataloader),
            desc="eval iteration",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            prediction_logits = model(input_ids, attention_masks)
            loss = loss_fn(prediction_logits, labels)
            losses.append(loss)

            prediction_classes = torch.argmax(prediction_logits, dim=1)
            tp, fp, tn, fn = calculate_stats(prediction_classes, labels)
            tp_total += tp
            fp_total += fp
            fn_total += fn

    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total != 0 else None
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total != 0 else None
    F1 = 2 * (precision * recall) / (precision + recall) if recall and precision else None
    avg_eval_loss = sum(losses) / len(losses)
    print(
        f"eval stats: avg loss {avg_eval_loss}, precision: {precision}, "
        f"recall: {recall}, F1: {F1}\n\n"
    )
    return F1


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    lr: float,
    save_path: Path,
    device: str,
):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    save_path.parent.mkdir(exist_ok=True)
    best_F1 = 0.0

    for _ in tqdm(range(n_epochs), total=n_epochs, desc="training the model"):
        train_iteration(model, train_dataloader, loss_fn, optimizer, device)
        F1 = eval_iteration(model, eval_dataloader, loss_fn, device)
        if F1 is not None and F1 > best_F1:
            best_F1 = F1
            torch.save(model.state_dict(), save_path)


def main(
    train_split_path: Path,
    eval_split_path: Path,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_path: Path,
    model_config: dict,
):
    device = get_available_torch_device()
    model_config["device"] = device
    model = BaselineBertDetector(**model_config)

    train_dataset = BaselineDataset(train_split_path, model.tokenizer, model.max_length)
    eval_dataset = BaselineDataset(eval_split_path, model.tokenizer, model.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    train(model, train_dataloader, eval_dataloader, n_epochs, lr, save_path, device)


if __name__ == "__main__":
    baseline_params = yaml.safe_load(open("params.yaml"))["models.baseline"]
    train_split_path = Path(baseline_params["train_split_path"])
    eval_split_path = Path(baseline_params["eval_split_path"])
    batch_size = baseline_params["batch_size"]
    n_epochs = int(baseline_params["n_epochs"])
    lr = float(baseline_params["lr"])
    save_path = Path(baseline_params["save_path"])
    model_config = yaml.safe_load(open("configs/model_configs.yaml"))["baseline"]
    main(
        train_split_path,
        eval_split_path,
        batch_size,
        n_epochs,
        lr,
        save_path,
        model_config,
    )
