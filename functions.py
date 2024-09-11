from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from classes import CustomDataset, TextClassifier
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


def preprocess_tsv(input_path: Path, output_path: Path) -> None:
    """
    指定されたTSVファイルを分類タスクの教師データに変換します。

    Parameters
    ----------
    input_path : Path
        アンケートの生データのTSVファイルのパス
    output_path : Path
        出力TSVファイルのパス
    """
    # TSVファイルを読み込み
    df = pd.read_csv(input_path, delimiter="\t")

    # 新しいデータフレームを作成
    processed_data = []
    questions = [
        ("設問2", "設問3"),
        ("設問4", "設問5"),
        ("設問6", "設問7"),
        ("設問8", "設問9"),
    ]
    satisfaction_labels = ["満足", "満足", "不満", "不満"]

    for (q1, q2), satisfaction in zip(questions, satisfaction_labels):
        for index, row in df.iterrows():
            processed_data.append([row[q1], satisfaction, row[q2], row["会員ID"]])

    # データフレームに変換
    processed_df = pd.DataFrame(
        processed_data, columns=["label", "satisfaction", "text", "id"]
    )

    # 欠損値の削除
    processed_df.dropna(inplace=True)

    # TSVファイルとして保存
    processed_df.to_csv(output_path, sep="\t", index=False)


def extract_long_text(input_path: Path, output_path: Path, min_length: int) -> None:
    """
    指定されたTSVファイルから特定の長さ以上の文章を抽出します。

    Parameters
    ----------
    input_path : Path
        前処理されたアンケートデータのTSVファイルのパス
    output_path : Path
        出力TSVファイルのパス
    min_length : int
        抽出する文章の最小の長さ
    """
    # TSVファイルを読み込み
    df = pd.read_csv(input_path, delimiter="\t")

    # 文字数が最小値以上の文章を抽出
    long_texts = df[df["text"].str.len() >= min_length]

    # TSVファイルとして保存
    long_texts.to_csv(output_path, sep="\t", index=False)


# データローダー作成関数
def create_dataloaders(
    texts: pd.Series,
    labels: pd.Series,
    tokenizer: BertTokenizer,
    batch_size: int,
    max_len: int,
) -> DataLoader:
    dataset = CustomDataset(texts.tolist(), labels.tolist(), tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size)


# 訓練関数
def train_model(
    classifier: TextClassifier,
    train_texts: pd.Series,
    val_texts: pd.Series,
    train_labels: pd.Series,
    val_labels: pd.Series,
    max_len: int,
    batch_size: int,
    lr: float,
    epochs: int,
) -> None:
    # オプティマイザーの初期化
    optimizer = AdamW(classifier.model.parameters(), lr=lr)

    # データローダー作成
    train_loader = create_dataloaders(
        train_texts, train_labels, classifier.tokenizer, batch_size, max_len
    )
    val_loader = create_dataloaders(
        val_texts, val_labels, classifier.tokenizer, batch_size, max_len
    )

    # 訓練と評価
    for epoch in range(epochs):
        train_acc, train_loss = classifier.train(train_loader, optimizer, epoch)
        val_acc = classifier.evaluate(val_loader)
        print(
            f"Epoch {epoch}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}"
        )
