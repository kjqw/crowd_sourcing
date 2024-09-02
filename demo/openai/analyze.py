# %%
import pickle
from collections import Counter
from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# %%
# モデルの選択
MODEL_NAME = "gpt-3.5-turbo"

# 学習データの最小文字数
# min_length = 10
min_length = 20

# パスの管理
data_path = Path("../bert/data")
input_data_path = data_path / f"data_long_texts_{min_length}.tsv"

# %%
# データの読み込み
data = pd.read_csv(input_data_path, sep="\t")

with open(f"data/df_compare_{MODEL_NAME}_{min_length}.pickle", "rb") as f:
    df_compare = pickle.load(f)

# %%
# ラベルのエンコード
label_encoder = LabelEncoder()
label_encoder.fit_transform(data["ラベル"])
num_labels = len(label_encoder.classes_)

# %%
df_compare

# %%
# ラベルごとのデータ数をカウント
label_counts = Counter(df_compare["label"])
# ラベルごとの一致数をカウント
label_correct_counts = Counter(df_compare[df_compare["is_match_label"]]["label"])
# ラベルが他のラベルに予測された数をカウント
label_misclassified_as_other_counts = Counter(
    df_compare[~df_compare["is_match_label"]]["label"]
)
# 他のラベルがラベルに予測された数をカウント
label_other_misclassified_as_label_counts = Counter(
    df_compare[~df_compare["is_match_label"]]["label_prediction"]
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
fig_sorted.savefig(f"data/LabelClassificationResult_{MODEL_NAME}_{min_length}.png")

# %%
# 正解率の計算
accuracy_label = df_compare["is_match_label"].mean()
accuracy_satisfaction = df_compare["is_match_satisfaction"].mean()


# 結果の表示
print(f"正解率(ラベル): {accuracy_label:.2%}")
print(f"正解率(満足度): {accuracy_satisfaction:.2%}")

# %%
# weighted-f1スコアの計算
from sklearn.metrics import f1_score

f1_label = f1_score(
    df_compare["label"], df_compare["label_prediction"], average="weighted"
)
f1_satisfaction = f1_score(
    df_compare["satisfaction"],
    df_compare["satisfaction_prediction"],
    average="weighted",
)

print(f"データ数: {len(df_compare)}")
print(f"weighted-f1(ラベル): {f1_label:.2%}")
print(f"weighted-f1(満足度): {f1_satisfaction:.2%}")
