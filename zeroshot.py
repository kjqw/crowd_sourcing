# %%
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm

# %%
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 128  # 最大トークン数
LR = 2e-5  # 学習率
TEST_SIZE = 0.2  # テストデータの割合
RANDOM_STATE = 0  # ランダムシード

# MODEL_NAME_BERT = "google-bert/bert-base-multilingual-cased"
MODEL_NAME_ZEROSHOT = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

DATA_PATH = Path(__file__).parent / "data" / "data_long_texts_10.tsv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# データを読み込む
df = pd.read_csv(DATA_PATH, delimiter="\t")

# %%
pipe = transformers.pipeline(
    "zero-shot-classification", model=MODEL_NAME_ZEROSHOT, device=DEVICE
)
# %%
labels_content = df["label"].unique().tolist()
labels_satisfaction = df["satisfaction"].unique().tolist()
texts = df["text"].tolist()

# %%
# バッチ処理のサイズを指定
BATCH_SIZE = 16  # 必要に応じて調整


# ゼロショット分類をバッチ処理で実行し、進捗を表示
def batch_process(
    texts: list[str],
    labels: list[str],
    pipe,
    batch_size: int = BATCH_SIZE,
    max_texts: int | None = None,
) -> list[str]:
    # 処理するテキスト数を制限
    if max_texts is not None:
        texts = texts[:max_texts]

    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
        batch_texts = texts[i : i + batch_size]
        results = pipe(batch_texts, candidate_labels=labels)
        # 最もスコアが高いラベルをバッチごとに取得
        batch_predictions = [result["labels"][0] for result in results]
        predictions.extend(batch_predictions)
    return predictions


# %%
# 処理するテキストの数を指定
MAX_TEXTS = None  # ここで処理したいテキストの数を指定（必要に応じて None に）

# 予測列の初期化
df["predicted_label_content"] = None
df["predicted_label_satisfaction"] = None

# 必要な行数だけ処理
df.loc[: MAX_TEXTS - 1, "predicted_label_content"] = batch_process(
    df["text"].tolist()[:MAX_TEXTS], labels_content, pipe, max_texts=MAX_TEXTS
)

df.loc[: MAX_TEXTS - 1, "predicted_label_satisfaction"] = batch_process(
    df["text"].tolist()[:MAX_TEXTS], labels_satisfaction, pipe, max_texts=MAX_TEXTS
)

# 一致しているかどうかを判定
df["is_correct_content"] = df["predicted_label_content"] == df["label"]
df["is_correct_satisfaction"] = df["predicted_label_satisfaction"] == df["satisfaction"]

# %%
df
# %%
