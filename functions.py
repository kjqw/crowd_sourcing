import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


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
        processed_data, columns=["ラベル", "満足度", "文章", "会員ID"]
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
    long_texts = df[df["文章"].str.len() >= min_length]

    # TSVファイルとして保存
    long_texts.to_csv(output_path, sep="\t", index=False)


def load_data(filepath: str) -> pd.DataFrame:
    """
    データを読み込み、DataFrameとして返す
    """
    return pd.read_csv(filepath, sep="\t")


def train_model(model, dataloader, optimizer, epochs: int = 3):
    """
    モデルを訓練する関数
    """
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # satisfactionは使わないので除外
            inputs = {
                key: val.to(model.device)
                for key, val in batch.items()
                if key not in ["satisfaction", "labels"]
            }
            labels = batch["labels"].to(model.device)

            optimizer.zero_grad()
            # `labels` を含む場合は以下の形に修正
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

    return model


def save_model(model, path: str):
    """
    モデルの重みを保存する関数
    """
    model.save_pretrained(path)


def evaluate_model(model, dataloader):
    """
    モデルを評価し、正答率を計算する関数
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                key: val.to(model.device)
                for key, val in batch.items()
                if key != "satisfaction"
            }
            labels = batch["labels"].to(model.device)
            outputs = model(**inputs)
            _, preds = torch.max(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy
