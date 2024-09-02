# %%
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# %%
# モデルの選択
# MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MODEL_NAME = "tohoku-nlp/bert-base-japanese-v3"

# 学習データの最小文字数
min_length = 10
# min_length = 20

# %%
# パスの管理
data_path = Path("data")
input_data_path = data_path / f"data_long_texts_{min_length}.tsv"
satisfaction_model_path = data_path / f"ModelSatisfaction_{MODEL_NAME.split("/")[-1]}_TextMinLength{min_length}"
label_model_path = data_path / f"ModelLabel_{MODEL_NAME.split("/")[-1]}_TextMinLength{min_length}"

# %%
print(satisfaction_model_path.name)
print(label_model_path.name)

# %%
# データが存在するかどうか
input_data_path.exists()

# %%
# データの読み込み
data = pd.read_csv(input_data_path, sep="\t")

# 満足度のエンコード（0: 不満, 1: 満足）
data["満足度"] = data["満足度"].map({"不満": 0, "満足": 1})

# ラベルのエンコード
label_encoder = LabelEncoder()
data["ラベル"] = label_encoder.fit_transform(data["ラベル"])
num_labels = len(label_encoder.classes_)

# デバイスの設定
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# モデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 満足度分類モデルのロード
satisfaction_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
satisfaction_model.to(device)  # モデルをデバイスに移動

# ラベル分類モデルのロード
label_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)
label_model.to(device)  # モデルをデバイスに移動

# %%
# データを訓練データとテストデータに分割
train_data, test_data = train_test_split(data, test_size=0.2)

# %%
# データセットクラスの作成
class CustomDataset(Dataset):
    """
    データセットクラス

    Parameters
    ----------
    encodings : dict
        エンコーディングされたデータ
    labels : list
        ラベルのリスト

    Returns
    -------
    item : dict
        エンコーディングされたデータとラベルのペア
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# データの前処理
def preprocess_data(data: pd.DataFrame, label_col: str) -> tuple:
    """
    データの前処理

    Parameters
    ----------
    data : pd.DataFrame
        データ
    label_col : str
        ラベルのカラム名

    Returns
    -------
    encodings : dict
        エンコーディングされたデータ
    labels : list
        ラベルのリスト
    """

    encodings = tokenizer(
        data["文章"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    labels = data[label_col].tolist()
    return encodings, labels

# %%
# 満足度分類用データセットの作成
satisfaction_encodings, satisfaction_labels = preprocess_data(train_data, "満足度")
satisfaction_dataset = CustomDataset(satisfaction_encodings, satisfaction_labels)

# ラベル分類用データセットの作成
label_encodings, label_labels = preprocess_data(train_data, "ラベル")
label_dataset = CustomDataset(label_encodings, label_labels)

# %%
# 評価指標の計算関数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# %%
# トレーニング引数の設定
train_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
)

# %%
# 満足度分類モデルのトレーナーの作成
satisfaction_trainer = Trainer(
    model=satisfaction_model,
    args=train_args,
    train_dataset=satisfaction_dataset,
    eval_dataset=satisfaction_dataset,
    compute_metrics=compute_metrics,
)

# %%
# 満足度分類モデルのトレーニング
satisfaction_trainer.train()

# %%
# モデルの保存
satisfaction_trainer.save_model(satisfaction_model_path)

# %%
# ラベル分類モデルのトレーナーの作成
label_trainer = Trainer(
    model=label_model,
    args=train_args,
    train_dataset=label_dataset,
    eval_dataset=label_dataset,
    compute_metrics=compute_metrics,
)

# %%
# ラベル分類モデルのトレーニング
label_trainer.train()

# %%
# モデルの保存
label_trainer.save_model(label_model_path)

# %%
# テストデータの準備
test_texts = test_data["文章"].tolist()
test_satisfactions = ["満足" if i == 1 else "不満" for i in test_data["満足度"]]
test_labels = test_data["ラベル"].tolist()

# テストデータのトークナイズ
test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, return_tensors="pt"
).to(device)

# モデルを評価モードに設定
satisfaction_model.eval()
label_model.eval()

# 予測の取得
with torch.no_grad():
    satisfaction_outputs = satisfaction_model(**test_encodings)
    satisfaction_preds = torch.argmax(satisfaction_outputs.logits, dim=1).cpu().numpy()

    label_outputs = label_model(**test_encodings)
    label_preds = torch.argmax(label_outputs.logits, dim=1).cpu().numpy()

# 数値から文字列に変換
label_result = label_encoder.inverse_transform(label_preds).tolist()
satisfaction_result = ["満足" if pred == 1 else "不満" for pred in satisfaction_preds]

# %%
df_compare_test_data = pd.DataFrame(
    {
        "満足度": test_satisfactions,
        "満足度予測": satisfaction_result,
        "満足度一致": [i == j for i, j in zip(test_satisfactions, satisfaction_result)],
        "ラベル": test_labels,
        "ラベル予測": label_result,
        "ラベル一致": [i == j for i, j in zip(test_labels, label_result)],
        "文章": test_texts,
    }
)

# %%
df_compare_test_data

# %%
# どの文章がテストデータであったかを記録しておく
test_data_path = (
    data_path / f"TestData_{MODEL_NAME.split('/')[-1]}_TextMinLength{min_length}.pickle"
)
with open(test_data_path, "wb") as f:
    pickle.dump(test_data, f)
