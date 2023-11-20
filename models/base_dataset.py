from pathlib import Path

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FakeNewsDetectorDataset(Dataset):
    def __init__(
        self,
        dataset_csv_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        include_logits: bool = False,
        attacker_tokenizer: PreTrainedTokenizer | None = None,
    ):
        """
        The last two parameters are meant to be used for training a generative attack on
        the model for which this dataset is intended.
        (I'm not a fan of this class design, maybe change it in the future)
        """
        super().__init__()
