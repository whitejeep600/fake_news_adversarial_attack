from pathlib import Path

import torch.nn
import yaml
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim import AdamW
from baseline.dataset import FakeNewsDataset
from baseline.model import BaselineBert
from src.torch_utils import get_device


def train_iteration(
    model: nn.Module, train_dataloader: DataLoader, loss_fn: _Loss, optimizer: Optimizer
) -> None:
    model.train()
    losses: list[float] = []
    for batch in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc="train iteration",
        leave=False,
    ):
        input_ids = batch["input_ids"].to(get_device())
        attention_masks = batch["attention_mask"].to(get_device())
        labels = batch["label"].to(get_device())

        optimizer.zero_grad()

        prediction_logits = model(input_ids, attention_masks)
        loss = loss_fn(prediction_logits, labels)
        losses.append(loss)

        loss.backward()
        optimizer.step()

    avg_train_loss = sum(losses) / len(losses)
    print(f"\naverage train loss: {avg_train_loss}")


def calculate_stats(
    predictions: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int, int, int]:
    tp = int(torch.sum(torch.logical_and(predictions, labels)).item())
    fp = int(
        torch.sum(torch.logical_and(predictions, torch.logical_not(labels))).item()
    )
    tn = int(
        torch.sum(
            torch.logical_and(
                torch.logical_not(predictions),
                torch.logical_not(labels),
            )
        ).item()
    )
    fn = int(
        torch.sum(torch.logical_and(torch.logical_not(predictions), labels)).item()
    )
    return tp, fp, tn, fn


def eval_iteration(
    model: nn.Module, eval_dataloader: DataLoader, loss_fn: _Loss, save_path: Path
) -> None:
    model.eval()
    losses: list[float] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    best_F1 = 0
    with torch.no_grad():
        for batch in tqdm(
            eval_dataloader,
            total=len(eval_dataloader),
            desc="eval iteration",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(get_device())
            attention_masks = batch["attention_mask"].to(get_device())
            labels = batch["label"].to(get_device())

            prediction_logits = model(input_ids, attention_masks)
            loss = loss_fn(prediction_logits, labels)
            losses.append(loss)

            prediction_classes = torch.argmax(prediction_logits, dim=1)
            tp, fp, tn, fn = calculate_stats(prediction_classes, labels)
            tp_total += tp
            fp_total += fp
            fn_total += fn

    precision = tp_total / (tp_total + fp_total)
    recall = tp_total / (tp_total + fn_total)
    F1 = 2 * (precision * recall) / (precision + recall)
    avg_eval_loss = sum(losses) / len(losses)
    print(
        f"eval stats: avg loss {avg_eval_loss}, precision: {precision}, "
        f"recall: {recall}, F1: {F1}\n\n"
    )
    if F1 > best_F1:
        best_F1 = F1
        torch.save(model.state_dict(), save_path)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    lr: float,
    save_path: Path,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    save_path.parent.mkdir(exist_ok=True)

    for _ in tqdm(range(n_epochs), total=n_epochs, desc="training the model"):
        train_iteration(model, train_dataloader, loss_fn, optimizer)
        eval_iteration(model, eval_dataloader, loss_fn, save_path)


def main(
    train_split_path: Path,
    eval_split_path: Path,
    bert_model_name: str,
    batch_size: int,
    max_length: int,
    n_epochs: int,
    lr: float,
    save_path: Path,
):
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    model = BaselineBert(bert_model_name, 2)
    model.to(get_device())

    train_dataset = FakeNewsDataset(train_split_path, tokenizer, max_length)
    eval_dataset = FakeNewsDataset(eval_split_path, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    train(model, train_dataloader, eval_dataloader, n_epochs, lr, save_path)


if __name__ == "__main__":
    baseline_params = yaml.safe_load(open("params.yaml"))["baseline"]
    bert_model_name = baseline_params["bert_model_name"]
    train_split_path = Path(baseline_params["train_split_path"])
    eval_split_path = Path(baseline_params["eval_split_path"])
    batch_size = baseline_params["batch_size"]
    max_length = int(baseline_params["max_length"])
    n_epochs = int(baseline_params["n_epochs"])
    lr = float(baseline_params["lr"])
    save_path = Path(baseline_params["save_path"])
    main(
        train_split_path,
        eval_split_path,
        bert_model_name,
        batch_size,
        max_length,
        n_epochs,
        lr,
        save_path,
    )
