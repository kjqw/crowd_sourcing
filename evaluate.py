# %%
from pathlib import Path

import pandas as pd
import torch
from classes import CustomDataset, TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader

# %%
# 設定: ハイパーパラメータとデバイス設定
BATCH_SIZE = 16
EPOCHS = 10
MAX_LEN = 128  # 最大トークン数
LR = 2e-5  # 学習率
TEST_SIZE = 0.2  # テストデータの割合
RANDOM_STATE = 0  # ランダムシード

MODEL_NAME_BERT = "google-bert/bert-base-multilingual-cased"
# MODEL_NAME_ZEROSHOT = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

DATA_PATH = Path(__file__).parent / "data" / "data_long_texts_10.tsv"
MODEL_SAVE_PATH = Path(__file__).parent / "saved_models" / "model_1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# データを読み込み、ラベルを数値に変換
df = pd.read_csv(DATA_PATH, delimiter="\t")

# LabelEncoderを使用してラベルを数値にエンコード
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

# 訓練データとテストデータの分割
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label_id"], test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# %%
# モデルの読み込み
classifier_content = TextClassifier(
    MODEL_NAME_BERT,
    num_labels=len(label_encoder.classes_),
    max_length=MAX_LEN,
    device=DEVICE,
    load_model_path=MODEL_SAVE_PATH,
)


# %%
# 予測関数の適用をまとめて行う
def predict_and_compare(row):
    prediction = classifier_content.predict_text(
        row["text"], label_encoder
    )  # textに対して予測を行う
    correct = prediction == row["original_label"]  # 正解かどうかを判定
    return pd.Series([prediction, correct], index=["predicted_label", "is_correct"])


# val_textsとval_labelsをDataFrameに変換
df = pd.DataFrame(
    {"text": val_texts, "original_label": label_encoder.inverse_transform(val_labels)}
)

# applyを使って一気に予測と正解判定を行い、新しいカラムを追加
df_result = df.join(df.apply(predict_and_compare, axis=1))

# %%
df_result

# %%
df_result["is_correct"].mean()
# %%
