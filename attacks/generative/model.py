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

    def parameters(self):
        return self.bert.parameters()

    @classmethod
    def from_config(cls, config: dict) -> "GenerativeAttacker":
        return GenerativeAttacker(config["model_name"], int(config["max_length"]))

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    def generate_with_greedy_decoding(
            self,
            inputs: torch.Tensor,
            max_length: int = 20
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Return a (sequence, scores) tuple where sequence is a tensor of shape (generation_len)
        containing the ids of the generated sequence, and scores is a list of len generation_len,
        whose each element is a tensor of shape [1, vocab_size] containing the predicted token
        logits for each step.

        """
        decoded = torch.Tensor([[self.bert.config.decoder_start_token_id]]).int()
        scores: list[torch.Tensor] = []
        for _ in range(max_length - 1):
            next_one = self.bert(
                input_ids=inputs,
                decoder_input_ids=decoded,
            )
            new_scores = next_one.logits[0][-1, :]
            next_id = torch.argmax(new_scores, dim=-1)
            decoded = torch.cat((decoded, torch.Tensor([[next_id]]).int()), dim=-1)
            scores.append(new_scores.unsqueeze(0))
            if next_id == self.bert.generation_config.eos_token_id:
                break
        return decoded.squeeze(0), scores

    # todo this shouldn't be defined like this with passing the reference model, fix
    def generate_ids_and_probabilities(
        self, batch: torch.Tensor, max_victim_length: int, reference_model: torch.nn.Module
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        torch.set_grad_enabled(True)
        generated_ids: list[torch.Tensor] = []
        token_probabilities: list[torch.Tensor] = []
        reference_probabilities: list = []
        for seq in batch:
            new_ids, scores = self.generate_with_greedy_decoding(seq.unsqueeze(0), 20)
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
                    reference_model.compute_transition_scores(
                        new_ids.unsqueeze(0), scores, normalize_logits=True
                    ).squeeze(0)
                )
            )
        all_token_ids = torch.stack(
            [
                pad(ids, (0, max_victim_length - len(ids)), "constant", PADDING)
                for ids in generated_ids
            ]
        )
        return all_token_ids, token_probabilities, reference_probabilities

    def decode_prefixes(self, generated_ids: torch.Tensor) -> list[list[str]]:
        results: list[list[str]] = []
        for i in range(len(generated_ids)):
            results.append([])
            decoded_length = -1
            for j in range(len(generated_ids[i])):
                prefix = self.tokenizer.batch_decode(
                    [generated_ids[i][: j + 1]], skip_special_tokens=True
                )[0]
                if len(prefix) == decoded_length:
                    break
                if len(prefix) != 0:
                    decoded_length = len(prefix)
                    results[-1].append(prefix)
        return results
