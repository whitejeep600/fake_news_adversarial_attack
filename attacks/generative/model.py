import torch
from torch.nn.functional import pad
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig

from attacks.base import AdversarialAttacker

PADDING = 1


class GenerativeAttacker(AdversarialAttacker):
    def __init__(self, model_name: str, max_length: int):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    @classmethod
    def from_config(cls, config: dict) -> "GenerativeAttacker":
        return GenerativeAttacker(config["model_name"], int(config["max_length"]))

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    # todo this shouldn't be defined like this with passing the reference model, fix
    def generate(
        self, batch: torch.Tensor, max_victim_length: int, reference_model: torch.nn.Module
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        # Return the generated ids, and a list of tensors of predicted token logits
        generated_ids: list[torch.Tensor] = []
        token_logits: list[torch.Tensor] = []
        scores: list = []
        for seq in batch:
            generation_config = GenerationConfig(
                return_dict_in_generate=True,
                output_scores=True,
                # min_length=int(0.7 * len(seq)),
                # todo debug max_length=min(int(1.3 * len(seq)), max_victim_length),
                max_length=20,
            )
            generation_output = self.bert.generate(
                seq.unsqueeze(0), generation_config=generation_config
            )
            new_ids = generation_output.sequences.squeeze(0)
            generated_ids.append(new_ids)
            token_logits.append(
                torch.Tensor(
                    [
                        generation_output.scores[i][0][new_ids[i + 1]]
                        for i in range(len(generation_output.scores))
                    ]
                )
            )
            scores.append(
                reference_model.compute_transition_scores(
                    generation_output.sequences, generation_output.scores
                ).squeeze(0)
            )
        all_token_ids = torch.stack(
            [
                pad(ids, (0, max_victim_length - len(ids)), "constant", PADDING)
                for ids in generated_ids
            ]
        )
        return all_token_ids, token_logits, scores
