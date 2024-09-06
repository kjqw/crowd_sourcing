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
        self.texts = data["文章"].tolist()

        # ラベルを数値化するマッピング
        self.label_mapping = {
            "医療・福祉": 0,
            "買物・飲食": 1,
            "住宅環境": 2,
            "移動・交通": 3,
            "遊び・娯楽": 4,
            "子育て": 5,
            "初等・中等教育": 6,
            "地域行政": 7,
            "デジタル生活": 8,
            "公共空間": 9,
            "都市景観": 10,
            "自然景観": 11,
            "自然の恵み": 12,
            "環境共生": 13,
            "自然災害": 14,
            "事故・犯罪": 15,
        }

        # ラベルを数値に変換
        self.labels = data["ラベル"].map(self.label_mapping).tolist()
        self.satisfaction = (
            data["満足度"].apply(lambda x: 1 if x == "満足" else 0).tolist()
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
