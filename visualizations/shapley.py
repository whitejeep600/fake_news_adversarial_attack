# An attempt to visualize the influence of every word in a given sample on the model's
# prediction. Takes a word's attention received from the CLS token as a proxy for
# influence. Prints the sample with words colored, the more red, the higher the
# attention. Requires a terminal supporting colored printing.

from pathlib import Path

import shap
import torch
import yaml
from tqdm import tqdm

from models.baseline.dataset import FakeNewsDataset
from src.evaluate_attack import MODELS_DICT
from src.torch_utils import get_available_torch_device


def main(model_class: str, weights_path: Path, eval_split_path: Path, plots_path: Path):
    plots_path.mkdir(exist_ok=True, parents=True)
    model_config = yaml.safe_load(open("configs/model_configs.yaml"))[model_class]
    device = get_available_torch_device()
    model_config["device"] = device
    model = MODELS_DICT[model_class](**model_config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    eval_dataset = FakeNewsDataset(eval_split_path, model.tokenizer, model.max_length)
    pipeline = model.to_pipeline()
    n_total = len(eval_dataset)
    for i, sample in enumerate(
        tqdm(
            eval_dataset.iterate_untokenized(),
            total=n_total,
            desc="Getting SHAP explanations for samples from the dataset",
        ),
    ):
        text = sample["text"]
        explainer = shap.Explainer(pipeline)
        shap_values = explainer([text])

        target_path = plots_path / f"explanation{i}.html"
        with open(target_path, "w") as file:
            file.write(shap.plots.text(shap_values, display=False))


if __name__ == "__main__":
    evaluation_params = yaml.safe_load(open("params.yaml"))["src.evaluate_attack"]
    visualization_params = yaml.safe_load(open("params.yaml"))["visualizations.cls_attention"]
    model_class = evaluation_params["model_class"]
    weights_path = Path(evaluation_params["weights_path"])
    eval_split_path = Path(evaluation_params["eval_split_path"])
    plots_path = Path(visualization_params["plots_path"])
    main(model_class, weights_path, eval_split_path, plots_path)
