# %%
import pickle
from collections import Counter
from pathlib import Path

import japanize_matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
DATA_PATH = Path(__file__).parent / "data" / "data_long_texts_10.tsv"

# %%
# データを読み込み、ラベルを数値に変換
df = pd.read_csv(DATA_PATH, delimiter="\t")

# %%
# 予測結果のデータを読み込む
with open(Path(__file__).parent / "data" / "df_result_content.pkl", "rb") as f:
    df_result_content = pickle.load(f)
with open(Path(__file__).parent / "data" / "df_result_satisfaction.pkl", "rb") as f:
    df_result_satisfaction = pickle.load(f)

# %%
df_result_content[df_result_content["is_correct"] == True]
# %%
df_result_content[df_result_content["is_correct"] == False]
# %%
df_result_satisfaction[df_result_satisfaction["is_correct"] == True]
# %%
df_result_satisfaction[df_result_satisfaction["is_correct"] == False]

# %%
# ラベルの分類結果の可視化
# df_compare = df_result_content
df_compare = df_result_satisfaction

# ラベルごとのデータ数をカウント
label_counts = Counter(df_compare["original_label"])
# ラベルごとの一致数をカウント
label_correct_counts = Counter(df_compare[df_compare["is_correct"]]["original_label"])

# ラベルごとの誤分類された数をカウント
label_misclassified_counts = Counter(
    df_compare[~df_compare["is_correct"]]["original_label"]
)

label_misclassified_counts
# %%
labels = sorted(label_counts.keys())

original_counts = [label_counts[label] for label in labels]
correct_counts = [label_correct_counts.get(label, 0) for label in labels]

misclassified_counts = [label_misclassified_counts.get(label, 0) for label in labels]

x = range(len(labels))

# プロットの作成
# fig, ax = plt.subplots(figsize=(10, 6))
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2

# フォントをTimes New Romanに設定
# plt.rcParams["font.family"] = "Times New Roman"

# Original label counts
ax.bar(
    x,
    original_counts,
    bar_width,
    label="Original Labels",
    align="center",
    color="tab:blue",
    hatch="//",
)

# Correctly classified counts
ax.bar(
    [i + bar_width for i in x],
    correct_counts,
    bar_width,
    label="Correct Predictions",
    align="center",
    color="tab:green",
    hatch="-",
)

# Misclassified counts
ax.bar(
    [i + 2 * bar_width for i in x],
    misclassified_counts,
    bar_width,
    label="Misclassified",
    align="center",
    color="tab:red",
    hatch="x",
)


ax.set_xlabel("Labels", fontsize=14)
ax.set_ylabel("Counts", fontsize=14)
# ax.set_title("ラベルごとの分類結果")
ax.set_xticks([i + 1.5 * bar_width for i in x])
ax.set_xticklabels(labels, rotation=90, fontsize=14)
ax.legend(fontsize=14)

plt.tight_layout()
# %%
