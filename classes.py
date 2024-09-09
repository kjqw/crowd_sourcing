import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextDataset(Dataset):
    """
    テキスト分類用のデータセットクラス
    """

    def __init__(
        self, data: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 128
    ):
        self.texts = data["text"].tolist()

        # ラベルを自動的に数値化
        self.label_to_id = {
            label: idx for idx, label in enumerate(data["label"].unique())
        }
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        # 数値化したラベルを保持
        self.labels = data["label"].map(self.label_to_id).tolist()
        self.satisfaction = (
            data["satisfaction"].apply(lambda x: 1 if x == "満足" else 0).tolist()
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        # テキストをトークナイズ
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # ラベルと満足度を付加
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["satisfaction"] = torch.tensor(self.satisfaction[idx])
        return item

    def get_label_mapping(self) -> dict:
        """
        ラベルと数値IDのマッピングを取得する
        """
        return self.label_to_id
