import torch


class FakeNewsDetector(torch.nn.Module):
    def get_label(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        prediction_logits = self(input_ids, attention_mask)
        return torch.argmax(prediction_logits, dim=1).item()
