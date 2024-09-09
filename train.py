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
# モデル初期化
classifier_content = TextClassifier(
    MODEL_NAME_BERT,
    num_labels=len(label_encoder.classes_),
    max_length=MAX_LEN,
    device=DEVICE,
)
optimizer = AdamW(classifier_content.model.parameters(), lr=LR)

# %%
# データセットの作成
train_dataset = CustomDataset(
    train_texts.tolist(), train_labels.tolist(), classifier_content.tokenizer, MAX_LEN
)
val_dataset = CustomDataset(
    val_texts.tolist(), val_labels.tolist(), classifier_content.tokenizer, MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# %%
# 訓練と評価
for epoch in range(EPOCHS):
    train_acc, train_loss = classifier_content.train(train_loader, optimizer, epoch)
    val_acc = classifier_content.evaluate(val_loader)
    print(
        f"Epoch {epoch}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}"
    )


# %%
# モデル保存
classifier_content.save_model(MODEL_SAVE_PATH)
# %%
