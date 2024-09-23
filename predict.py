# %%
import json
import pickle
from pathlib import Path

import functions
import pandas as pd
import torch
from classes import TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
MODEL_NAME_BERT = "google-bert/bert-base-multilingual-cased"  # 事前学習済みモデル
MODEL_NUM = 10  # モデルの数
DATA_PATH = (
    Path(__file__).parent / "data" / "data_long_texts_10.tsv"
)  # 教師データのパス
MODEL_PATH = Path(__file__).parent / "models"  # 学習済みモデルの保存先
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # デバイス

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

for i in range(MODEL_NUM):
    # ハイパーパラメータの読み込み
    with open(MODEL_PATH / f"hyperparameters_{i}.json", "r") as f:
        hyperparameters = json.load(f)
    random_state = hyperparameters["RANDOM_STATE"]
    test_size = hyperparameters["TEST_SIZE"]
    max_len = hyperparameters["MAX_LEN"]

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
        test_size=test_size,
        random_state=random_state,
    )

    # モデルの読み込み
    classifier_content = TextClassifier(
        MODEL_NAME_BERT,
        num_labels=len(label_encoder_content.classes_),
        max_length=max_len,
        device=DEVICE,
        load_model_path=MODEL_PATH / f"content_model_{i}",
    )
    classifier_satisfaction = TextClassifier(
        MODEL_NAME_BERT,
        num_labels=len(label_encoder_satisfaction.classes_),
        max_length=max_len,
        device=DEVICE,
        load_model_path=MODEL_PATH / f"satisfaction_model_{i}",
    )

    # val_textsとval_labelsをDataFrameに変換
    df_content = pd.DataFrame(
        {
            "text": val_texts,
            "original_label": label_encoder_content.inverse_transform(
                val_labels_content
            ),
        }
    )
    df_satisfaction = pd.DataFrame(
        {
            "text": val_texts,
            "original_label": label_encoder_satisfaction.inverse_transform(
                val_labels_satisfaction
            ),
        }
    )

    # classifier_contentとlabel_encoder_contentを使って予測と正解判定
    df_result_content = df_content.join(
        df_content.apply(
            lambda row: functions.predict_and_compare(
                row, classifier_content, label_encoder_content
            ),
            axis=1,
        )
    )

    # classifier_satisfactionとlabel_encoder_satisfactionを使って予測と正解判定
    df_result_satisfaction = df_satisfaction.join(
        df_satisfaction.apply(
            lambda row: functions.predict_and_compare(
                row, classifier_satisfaction, label_encoder_satisfaction
            ),
            axis=1,
        )
    )

    # 結果を保存
    with open(Path(__file__).parent / "data" / f"df_result_content_{i}.pkl", "wb") as f:
        pickle.dump(df_result_content, f)

    with open(
        Path(__file__).parent / "data" / f"df_result_satisfaction_{i}.pkl", "wb"
    ) as f:
        pickle.dump(df_result_satisfaction, f)

    # GPUのメモリを解放
    del classifier_content, classifier_satisfaction
    torch.cuda.empty_cache()

    # メモリを解放
    del (
        train_texts,
        val_texts,
        train_labels_content,
        val_labels_content,
        train_labels_satisfaction,
        val_labels_satisfaction,
        df_content,
        df_satisfaction,
        df_result_content,
        df_result_satisfaction,
    )

# %%
