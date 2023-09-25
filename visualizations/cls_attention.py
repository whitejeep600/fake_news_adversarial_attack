# An attempt to visualize the influence of every word in a given sample on the model's
# prediction. Takes a word's attention received from the CLS token as a proxy for
# influence. Prints the sample with words colored, the more red, the higher the
# attention. Requires a terminal supporting colored printing.

from pathlib import Path

import torch
import truecolor
import yaml

from models.baseline.dataset import FakeNewsDataset
from src.evaluate_attack import MODELS_DICT
from src.torch_utils import get_available_torch_device

if __name__ == '__main__':
    evaluation_params = yaml.safe_load(open("params.yaml"))["src.evaluate_attack"]
    model_class = evaluation_params["model_class"]
    model_config = yaml.safe_load(open("configs/model_configs.yaml"))[model_class]
    device = get_available_torch_device()
    model_config["device"] = device
    weights_path = Path(evaluation_params["weights_path"])
    model = MODELS_DICT[model_class](**model_config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    eval_split_path = Path(evaluation_params["eval_split_path"])
    eval_dataset = FakeNewsDataset(eval_split_path, model.tokenizer, model.max_length)
    for sample in eval_dataset.iterate_untokenized():
        text = sample["text"]
        tokenized_text = model.tokenizer(
            text,
            return_tensors="pt",
            max_length=model.max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized_text["input_ids"].to(model.device)
        attention_mask = tokenized_text["attention_mask"].to(model.device)
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask,
                             output_attentions=True)
        tokenized_length = int(torch.sum(input_ids != 0).item())

        last_layer_attentions = outputs.attentions[-1][0]

        average_head_attentions = torch.mean(last_layer_attentions, axis=0)

        truncated_attentions = \
            average_head_attentions[:tokenized_length, :tokenized_length]

        attentions_from_cls_token = truncated_attentions[0, :]

        importance_scores = attentions_from_cls_token
        importance_scores /= torch.max(importance_scores).item()
        assert all(0 <= importance_scores)
        assert all(importance_scores <= 1)
        importance_scores /= torch.max(importance_scores).item()
        red_intensities = [int(score * 255) for score in importance_scores]

        tokenized_input = \
            model.tokenizer.convert_ids_to_tokens(input_ids[0])[:tokenized_length]

        colored_prints = [
            truecolor.color_text(
                tokenized_input[i], foreground=(red_intensities[i], 128, 128)
            )
            for i in range(len(tokenized_input))
        ]
        colored = " ".join(colored_prints)
        print(colored)
        print("\n")
