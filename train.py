# %%
import json
import random
from pathlib import Path

import functions
import pandas as pd
import torch
from classes import TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# %%
TRAINING_REPEATS = 10  # 訓練の繰り返し回数
EPOCHS = 5  # 一訓練あたりのエポック数
BATCH_SIZE = 16  # バッチサイズ
MAX_LEN = 128  # 最大トークン数
LR = 2e-5  # 学習率
TEST_SIZE = 0.2  # テストデータの割合
DATA_PATH = (
    Path(__file__).parent / "data" / "data_long_texts_10.tsv"
)  # 教師データのパス
MAX_RANDOM_SEED = 100000  # 乱数の最大値
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # デバイス

MODEL_NAME_BERT = "google-bert/bert-base-multilingual-cased"  # 事前学習済みモデル

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
for i in range(TRAINING_REPEATS):
    # 訓練データとテストデータの分割
    RANDOM_STATE = random.randint(0, MAX_RANDOM_SEED)
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

    # モデル初期化
    classifier_content = TextClassifier(
        MODEL_NAME_BERT,
        num_labels=len(label_encoder_content.classes_),
        max_length=MAX_LEN,
        device=DEVICE,
    )

    # 訓練
    tqdm.write(f"内容分類モデルの訓練 {i + 1}/{TRAINING_REPEATS}")
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

    # モデル保存
    save_path_content_model = Path(__file__).parent / "models" / f"content_model_{i}"
    classifier_content.save_model(save_path_content_model)

    # GPUのメモリを解放
    del classifier_content
    torch.cuda.empty_cache()

    # モデル初期化
    classifier_satisfaction = TextClassifier(
        MODEL_NAME_BERT,
        num_labels=len(label_encoder_satisfaction.classes_),
        max_length=MAX_LEN,
        device=DEVICE,
    )

    # 訓練
    tqdm.write(f"満足度分類モデルの訓練 {i + 1}/{TRAINING_REPEATS}")
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

    # モデル保存
    save_path_satisfaction_model = (
        Path(__file__).parent / "models" / f"satisfaction_model_{i}"
    )
    classifier_satisfaction.save_model(save_path_satisfaction_model)

    # 訓練に使用したハイパーパラメータを保存
    with open(Path(__file__).parent / "models" / f"hyperparameters_{i}.json", "w") as f:
        json.dump(
            {
                "RANDOM_STATE": RANDOM_STATE,
                "TEST_SIZE": TEST_SIZE,
                "MAX_LEN": MAX_LEN,
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "LR": LR,
            },
            f,
        )

    # GPUのメモリを解放
    del classifier_satisfaction
    torch.cuda.empty_cache()

    # メモリを解放
    del (
        train_texts,
        val_texts,
        train_labels_content,
        val_labels_content,
        train_labels_satisfaction,
        val_labels_satisfaction,
    )

# %%
