# %%
import pickle
from collections import Counter
from pathlib import Path

import japanize_matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
# 予測結果のデータを読み込む
with open(Path(__file__).parent / "data" / "df_result_content_tf.pkl", "rb") as f:
    df_result_content_tf = pickle.load(f)
with open(Path(__file__).parent / "data" / "df_result_satisfaction_tf.pkl", "rb") as f:
    df_result_satisfaction_tf = pickle.load(f)

# %%
dfs = [df_result_content_tf, df_result_satisfaction_tf]
figs = []
for df in dfs:
    labels = df.index.tolist()
    x = range(len(labels))

    # フォントをTimes New Romanに設定
    # plt.rcParams["font.family"] = "Times New Roman"

    # プロットの作成
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2

    # ラベルの数
    ax.bar(
        x,
        df["total_count"],
        bar_width,
        label="Original Labels",
        align="center",
        color="tab:blue",
        hatch="//",
    )

    # 正解数
    ax.bar(
        [i + bar_width for i in x],
        df["true_count"],
        bar_width,
        label="Correct Predictions",
        align="center",
        color="tab:green",
        hatch="-",
    )

    # 誤り数
    ax.bar(
        [i + 2 * bar_width for i in x],
        df["false_count"],
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
    figs.append(fig)

    plt.close()


# %%
save_path_content = Path(__file__).parent / "data/images" / "result_content.png"
save_path_satisfaction = (
    Path(__file__).parent / "data/images" / "result_satisfaction.png"
)
figs[0].savefig(save_path_content)
figs[1].savefig(save_path_satisfaction)

# %%
