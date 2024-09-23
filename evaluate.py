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
    with open(Path(__file__).parent / "data" / f"df_result_content_{i}.pkl", "rb") as f:
        df_results_content.append(pickle.load(f))
    with open(
        Path(__file__).parent / "data" / f"df_result_satisfaction_{i}.pkl", "rb"
    ) as f:
        df_results_satisfaction.append(pickle.load(f))

# %%
# 最初のデータフレームのtextとoriginal_labelを基準に他のデータフレームを結合
df_result_content = df_results_content[0][["text", "original_label"]].copy()

# 他のデータフレームからpredicted_labelとis_correctを追加していく
for i in range(MODEL_NUM):
    df_result_content = df_result_content.join(
        df_results_content[i][["predicted_label", "is_correct"]].rename(
            columns={
                "predicted_label": f"predicted_label_{i+1}",
                "is_correct": f"is_correct_{i+1}",
            }
        ),
        how="outer",
    )

# is_correctのTrueとFalseをカウントする新しい列を作成
df_result_content["true_count"] = df_result_content.filter(like="is_correct").apply(
    lambda row: (row == True).sum(), axis=1
)
df_result_content["false_count"] = df_result_content.filter(like="is_correct").apply(
    lambda row: (row == False).sum(), axis=1
)

# 結合された結果を表示
df_result_content


# %%
df_result_content["true_count"].sum(), df_result_content["false_count"].sum()
# %%
# 全indexを取得
indices_0 = df_results_content[0].index.tolist()
indices_1 = df_results_content[1].index.tolist()

common_indices = list(set(indices_0) & set(indices_1))
common_indices

# %%
# indexがnのデータを表示
n = 1024
print(df_results_content[0].loc[n])
print(df_results_content[1].loc[n])
# %%

# # %%
# # 正解率を集計
# correct_rates_content_all = []
# correct_rates_satisfaction_all = []
# correct_rates_content_groupyby = []
# correct_rates_satisfaction_groupby = []
# for i in range(MODEL_NUM):
#     correct_rates_content_all.append(df_results_content[i]["is_correct"].mean())
#     correct_rates_content_groupyby.append(
#         df_results_content[i].groupby("original_label")["is_correct"].mean()
#     )
#     correct_rates_satisfaction_all.append(
#         df_results_satisfaction[i]["is_correct"].mean()
#     )
#     correct_rates_satisfaction_groupby.append(
#         df_results_satisfaction[i].groupby("original_label")["is_correct"].mean()
#     )

# # %%
# correct_rates_content_groupyby[0]
# # %%
# # 正解率の平均と分散を計算
# correct_rates_content_all = pd.Series(correct_rates_content_all)
# correct_rates_satisfaction_all = pd.Series(correct_rates_satisfaction_all)
# correct_rates_content_groupyby = pd.DataFrame(correct_rates_content_groupyby).T
# correct_rates_satisfaction_groupby = pd.DataFrame(correct_rates_satisfaction_groupby).T
# correct_rates_content_all_mean = correct_rates_content_all.mean()
# correct_rates_satisfaction_all_mean = correct_rates_satisfaction_all.mean()
# correct_rates_content_all_std = correct_rates_content_all.std()
# correct_rates_satisfaction_all_std = correct_rates_satisfaction_all.std()
# correct_rates_content_groupyby_mean = correct_rates_content_groupyby.mean()
# correct_rates_satisfaction_groupby_mean = correct_rates_satisfaction_groupby.mean()
# correct_rates_content_groupyby_std = correct_rates_content_groupyby.std()
# correct_rates_satisfaction_groupby_std = correct_rates_satisfaction_groupby.std()

# # %%
# # 結果を出力
# print("Content Classification")
# print(f"Mean: {correct_rates_content_all_mean:.2f}")
# print(f"Std: {correct_rates_content_all_std:.2f}")
# print()
# print("Satisfaction Classification")
# print(f"Mean: {correct_rates_satisfaction_all_mean:.2f}")
# print(f"Std: {correct_rates_satisfaction_all_std:.2f}")
# print()
# print("Content Classification Groupby")
# print(correct_rates_content_groupyby_mean)
# print(correct_rates_content_groupyby_std)
# print()
# print("Satisfaction Classification Groupby")
# print(correct_rates_satisfaction_groupby_mean)
# print(correct_rates_satisfaction_groupby_std)
# # %%
