from pathlib import Path

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FakeNewsDetectorDataset(Dataset):
    def __init__(self, dataset_csv_path: Path, tokenizer: PreTrainedTokenizer, max_length: int):
        super().__init__()
