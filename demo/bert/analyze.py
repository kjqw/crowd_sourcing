# %%
import pickle
from collections import Counter
from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

# %%
# モデルの選択
# MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MODEL_NAME = "tohoku-nlp/bert-base-japanese-v3"

# 学習データの最小文字数
min_length = 10
# min_length = 20

# パスの管理
data_path = Path("data")
input_data_path = data_path / f"data_long_texts_{min_length}.tsv"
satisfaction_model_path = data_path / f"ModelSatisfaction_{MODEL_NAME.split("/")[-1]}_TextMinLength{min_length}"
label_model_path = data_path / f"ModelLabel_{MODEL_NAME.split("/")[-1]}_TextMinLength{min_length}"

# %%
# データの読み込み
data = pd.read_csv(input_data_path, sep="\t")

# デバイスの設定
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# ラベルのエンコード
label_encoder = LabelEncoder()
label_encoder.fit_transform(data["ラベル"])
num_labels = len(label_encoder.classes_)

# %%
# モデルとトークナイザーのロード

# トークナイザーのロード
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 満足度分類モデルのロード
satisfaction_model = BertForSequenceClassification.from_pretrained(
    satisfaction_model_path
)
satisfaction_model.to(device)  # モデルをデバイスに移動

# ラベル分類モデルのロード
label_model = BertForSequenceClassification.from_pretrained(label_model_path)
label_model.to(device)  # モデルをデバイスに移動

# %%
# 評価したいテキストの読み込み

use_all_data_for_evaluation = True  # Trueの場合、すべてのデータを使用して評価を行う。 Falseの場合、テストデータのみを使用する。
use_all_data_for_evaluation = False

if use_all_data_for_evaluation:
    texts_df = data
    texts = texts_df["文章"].tolist()
else:
    with open(
        data_path
        / f"TestData_{MODEL_NAME.split('/')[-1]}_TextMinLength{min_length}.pickle",
        "rb",
    ) as f:
        texts_df = pickle.load(f)
    texts_df = texts_df.sort_index()  # インデックスの整合性を保つためにソート
    texts_df["満足度"] = ["満足" if i == 1 else "不満" for i in texts_df["満足度"]] # ラベルを文字列に変換
    texts_df["ラベル"] = label_encoder.inverse_transform(texts_df["ラベル"]) # ラベルを文字列に変換
    texts = texts_df["文章"].tolist()

# %%
texts_df

# %%
# 学習済みモデルで分類
with torch.no_grad():
    # 満足度の予測
    satisfaction_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    satisfaction_texts = satisfaction_texts.to(device)
    satisfaction_outputs = satisfaction_model(**satisfaction_texts)
    satisfaction_predictions = torch.argmax(satisfaction_outputs.logits, dim=1).cpu().numpy()

    # ラベルの予測
    label_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    label_texts = label_texts.to(device)
    label_outputs = label_model(**label_texts)
    label_predictions = torch.argmax(label_outputs.logits, dim=1).cpu().numpy()

# %%
# gpuのメモリ解放
torch.cuda.empty_cache()

# %%
# 数値から文字列に変換
satisfaction_predictions_str = [
    "満足" if i == 1 else "不満" for i in satisfaction_predictions
]
label_predictions_str = label_encoder.inverse_transform(label_predictions).tolist()

# %%
# 予測との比較
df_compare = pd.DataFrame(
    {
        "満足度": texts_df["満足度"],
        "満足度予測": satisfaction_predictions_str,
        "満足度一致": texts_df["満足度"] == satisfaction_predictions_str,
        "ラベル": texts_df["ラベル"],
        "ラベル予測": label_predictions_str,
        "ラベル一致": texts_df["ラベル"] == label_predictions_str,
        "文章": texts,
    }
)

# %%
df_compare

# %%
# 比較結果を保存
with open(
    data_path
    / f"df_compare_{MODEL_NAME.split('/')[-1]}_TextMinLength{min_length}_{'AllData' if use_all_data_for_evaluation else 'TestData'}.pickle",
    "wb",
) as f:
    pickle.dump(df_compare, f)

# %%
# ラベルの分類結果の可視化

# ラベルごとのデータ数をカウント
label_counts = Counter(df_compare["ラベル"])
# ラベルごとの一致数をカウント
label_correct_counts = Counter(df_compare[df_compare["ラベル一致"]]["ラベル"])
# ラベルが他のラベルに予測された数をカウント
label_misclassified_as_other_counts = Counter(
    df_compare[~df_compare["ラベル一致"]]["ラベル"]
)
# 他のラベルがラベルに予測された数をカウント
label_other_misclassified_as_label_counts = Counter(
    df_compare[~df_compare["ラベル一致"]]["ラベル予測"]
)

labels = sorted(label_counts.keys())

original_counts = [label_counts[label] for label in labels]
correct_counts = [label_correct_counts.get(label, 0) for label in labels]
misclassified_as_other_counts = [
    label_misclassified_as_other_counts.get(label, 0) for label in labels
]
other_misclassified_as_label_counts = [
    label_other_misclassified_as_label_counts.get(label, 0) for label in labels
]

x = range(len(labels))

# プロットの作成
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2

# もともとのラベルのデータ数
ax.bar(
    x,
    original_counts,
    bar_width,
    label="元のラベル数",
    align="center",
    color="tab:blue",
    hatch="//",
)
# もとのラベルとラベル予測が一致したデータ数
ax.bar(
    [i + bar_width for i in x],
    correct_counts,
    bar_width,
    label="予測の一致数",
    align="center",
    color="tab:green",
    hatch="-",
)
# もとのラベルに他のラベルが予測されたデータ数
ax.bar(
    [i + 2 * bar_width for i in x],
    misclassified_as_other_counts,
    bar_width,
    label="他のラベルとして予測された数",
    align="center",
    color="tab:red",
    hatch="+",
)
# 他のラベルがラベルに予測されたデータ数
ax.bar(
    [i + 3 * bar_width for i in x],
    other_misclassified_as_label_counts,
    bar_width,
    label="他のラベルから予測された数",
    align="center",
    color="tab:orange",
    hatch="x",
)

ax.set_xlabel("ラベル")
ax.set_ylabel("件数")
# ax.set_title("ラベルごとの分類結果")
ax.set_xticks([i + 1.5 * bar_width for i in x])
ax.set_xticklabels(labels, rotation=90)
ax.legend()

plt.tight_layout()

# %%
fig_path = (
    data_path
    / f"LabelClassificationResult_{MODEL_NAME.split('/')[-1]}_TextMinLength{min_length}_{'AllData' if use_all_data_for_evaluation else 'TestData'}.png"
)
fig.savefig(fig_path)

# %%
# ラベルの分類結果の可視化（ラベルごとにソート）

# ラベルを元のラベル数でソート
sorted_labels = sorted(
    label_counts.keys(), key=lambda label: label_counts[label], reverse=True
)

sorted_original_counts = [label_counts[label] for label in sorted_labels]
sorted_correct_counts = [label_correct_counts.get(label, 0) for label in sorted_labels]
sorted_misclassified_as_other_counts = [
    label_misclassified_as_other_counts.get(label, 0) for label in sorted_labels
]
sorted_other_misclassified_as_label_counts = [
    label_other_misclassified_as_label_counts.get(label, 0) for label in sorted_labels
]

x = range(len(sorted_labels))

# プロットの作成
fig_sorted, ax_sorted = plt.subplots(figsize=(10, 6))

bar_width = 0.2

# もともとのラベルのデータ数
ax_sorted.bar(
    x,
    sorted_original_counts,
    bar_width,
    label="元のラベル数",
    align="center",
    color="tab:blue",
    hatch="//",
)
# もとのラベルとラベル予測が一致したデータ数
ax_sorted.bar(
    [i + bar_width for i in x],
    sorted_correct_counts,
    bar_width,
    label="予測の一致数",
    align="center",
    color="tab:green",
    hatch="-",
)
# もとのラベルに他のラベルが予測されたデータ数
ax_sorted.bar(
    [i + 2 * bar_width for i in x],
    sorted_misclassified_as_other_counts,
    bar_width,
    label="他のラベルとして予測された数",
    align="center",
    color="tab:red",
    hatch="+",
)
# 他のラベルがラベルに予測されたデータ数
ax_sorted.bar(
    [i + 3 * bar_width for i in x],
    sorted_other_misclassified_as_label_counts,
    bar_width,
    label="他のラベルから予測された数",
    align="center",
    color="tab:orange",
    hatch="x",
)

ax_sorted.set_xlabel("ラベル")
ax_sorted.set_ylabel("件数")
# ax_sorted.set_title("ラベルごとの分類結果")
ax_sorted.set_xticks([i + 1.5 * bar_width for i in x])
ax_sorted.set_xticklabels(sorted_labels, rotation=90)
ax_sorted.legend()

plt.tight_layout()
plt.show()

# %%
fig_sorted_path = (
    data_path
    / f"LabelClassificationResultSorted_{MODEL_NAME.split('/')[-1]}_TextMinLength{min_length}_{'AllData' if use_all_data_for_evaluation else 'TestData'}.png"
)
fig_sorted.savefig(fig_sorted_path)
