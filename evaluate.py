# %%
import pickle
from pathlib import Path

import pandas as pd

# %%
MODEL_NUM = 10  # モデルの数

# %%
# 結果を読み込み
df_results_content = []
df_results_satisfaction = []
for i in range(MODEL_NUM):
    with open(
        Path(__file__).parent / "data/pickle" / f"df_result_content_{i}.pkl", "rb"
    ) as f:
        df_results_content.append(pickle.load(f))
    with open(
        Path(__file__).parent / "data/pickle" / f"df_result_satisfaction_{i}.pkl", "rb"
    ) as f:
        df_results_satisfaction.append(pickle.load(f))


# %%
# 複数の結果を1つのデータフレームに結合
df_result_content = df_results_content[0][["text", "original_label"]].copy()
for i in range(MODEL_NUM):
    df_result_content = pd.merge(
        df_result_content,
        df_results_content[i][
            ["text", "original_label", "predicted_label", "is_correct"]
        ].rename(
            columns={
                "predicted_label": f"predicted_label_{i+1}",
                "is_correct": f"is_correct_{i+1}",
            }
        ),
        on=["text", "original_label"],
        how="outer",
    )

# is_correctのTrueとFalseをカウントする新しい列を作成
df_result_content["true_count"] = df_result_content.filter(like="is_correct").apply(
    lambda row: (row == True).sum(), axis=1
)
df_result_content["false_count"] = df_result_content.filter(like="is_correct").apply(
    lambda row: (row == False).sum(), axis=1
)

# %%
# 結合された結果を表示
df_result_content

# %%
df_result_satisfaction = df_results_satisfaction[0][["text", "original_label"]].copy()
for i in range(MODEL_NUM):
    df_result_satisfaction = pd.merge(
        df_result_satisfaction,
        df_results_satisfaction[i][
            ["text", "original_label", "predicted_label", "is_correct"]
        ].rename(
            columns={
                "predicted_label": f"predicted_label_{i+1}",
                "is_correct": f"is_correct_{i+1}",
            }
        ),
        on=["text", "original_label"],
        how="outer",
    )

# is_correctのTrueとFalseをカウントする新しい列を作成
df_result_satisfaction["true_count"] = df_result_satisfaction.filter(
    like="is_correct"
).apply(lambda row: (row == True).sum(), axis=1)
df_result_satisfaction["false_count"] = df_result_satisfaction.filter(
    like="is_correct"
).apply(lambda row: (row == False).sum(), axis=1)

# %%
df_result_satisfaction

# %%
# 元のラベルごとに正解数と不正解数を集計
df_result_content_tf = df_result_content[
    ["original_label", "true_count", "false_count"]
].copy()
df_result_satisfaction_tf = df_result_satisfaction[
    ["original_label", "true_count", "false_count"]
].copy()

df_result_content_tf = df_result_content_tf.groupby("original_label").sum()
df_result_satisfaction_tf = df_result_satisfaction_tf.groupby("original_label").sum()

# %%
# ラベルの合計数を列に追加
df_result_content_tf["total_count"] = (
    df_result_content_tf["true_count"] + df_result_content_tf["false_count"]
)
df_result_satisfaction_tf["total_count"] = (
    df_result_satisfaction_tf["true_count"] + df_result_satisfaction_tf["false_count"]
)

# ラベルの合計数の列を左に移動
df_result_content_tf = df_result_content_tf[
    ["total_count", "true_count", "false_count"]
].copy()
df_result_satisfaction_tf = df_result_satisfaction_tf[
    ["total_count", "true_count", "false_count"]
].copy()

# 正解率を列に追加
df_result_content_tf["accuracy"] = df_result_content_tf["true_count"] / (
    df_result_content_tf["true_count"] + df_result_content_tf["false_count"]
)
df_result_satisfaction_tf["accuracy"] = df_result_satisfaction_tf["true_count"] / (
    df_result_satisfaction_tf["true_count"] + df_result_satisfaction_tf["false_count"]
)

# %%
df_result_content_tf
# %%
df_result_satisfaction_tf
# %%
print(
    f"内容予測の全体の正解率: {df_result_content_tf['true_count'].sum() /(df_result_content_tf['true_count'].sum() + df_result_content_tf['false_count'].sum())}"
)
print(
    f"満足度予測の全体の正解率: {df_result_satisfaction_tf['true_count'].sum() /(df_result_satisfaction_tf['true_count'].sum() + df_result_satisfaction_tf['false_count'].sum())}"
)

# %%
# 結果を保存
with open(
    Path(__file__).parent / "data/pickle" / "df_result_content_tf.pkl", "wb"
) as f:
    pickle.dump(df_result_content_tf, f)
with open(
    Path(__file__).parent / "data/pickle" / "df_result_satisfaction_tf.pkl", "wb"
) as f:
    pickle.dump(df_result_satisfaction_tf, f)
# %%
