# %%
from pathlib import Path

import functions
import pandas as pd
import torch
from classes import CustomDataset, TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader

# %%
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 128  # 最大トークン数
LR = 2e-5  # 学習率
TEST_SIZE = 0.2  # テストデータの割合
RANDOM_STATE = 0  # ランダムシード

MODEL_NAME_BERT = "google-bert/bert-base-multilingual-cased"
# MODEL_NAME_ZEROSHOT = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

DATA_PATH = Path(__file__).parent / "data" / "data_long_texts_10.tsv"
SAVE_PATH_CONTENT_MODEL = Path(__file__).parent / "saved_models" / "model_content_1"
SAVE_PATH_SATISFACTION_MODEL = (
    Path(__file__).parent / "saved_models" / "model_satisfaction_1"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# データを読み込み、ラベルを数値に変換
df = pd.read_csv(DATA_PATH, delimiter="\t")

# LabelEncoderを使用してラベルを数値にエンコード
label_encoder_content = LabelEncoder()
label_encoder_satisfaction = LabelEncoder()
df["label_id_content"] = label_encoder_content.fit_transform(df["label"])
df["label_id_satisfaction"] = label_encoder_satisfaction.fit_transform(
    df["satisfaction"]
)

# %%
# 訓練データとテストデータの分割
(
    train_texts,
    val_texts,
    train_labels_content,
    val_labels_content,
    train_labels_satisfaction,
    val_labels_satisfaction,
) = train_test_split(
    df["text"],
    df["label_id_content"],
    df["label_id_satisfaction"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# %%
# モデル初期化
classifier_content = TextClassifier(
    MODEL_NAME_BERT,
    num_labels=len(label_encoder_content.classes_),
    max_length=MAX_LEN,
    device=DEVICE,
)

# %%
# 訓練
functions.train_model(
    classifier_content,
    train_texts,
    val_texts,
    train_labels_content,
    val_labels_content,
    MAX_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
)

# %%
# モデル保存
classifier_content.save_model(SAVE_PATH_CONTENT_MODEL)

# %%
# 変数を削除
del classifier_content

# %%
# GPUのメモリを解放
torch.cuda.empty_cache()

# %%
# モデル初期化
classifier_satisfaction = TextClassifier(
    MODEL_NAME_BERT,
    num_labels=len(label_encoder_satisfaction.classes_),
    max_length=MAX_LEN,
    device=DEVICE,
)

# %%
# 訓練
functions.train_model(
    classifier_satisfaction,
    train_texts,
    val_texts,
    train_labels_satisfaction,
    val_labels_satisfaction,
    MAX_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
)

# %%
# モデル保存
classifier_satisfaction.save_model(SAVE_PATH_SATISFACTION_MODEL)

# %%
# 変数を削除
del classifier_satisfaction

# %%
# GPUのメモリを解放
torch.cuda.empty_cache()

# %%
