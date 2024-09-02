# %%
import pickle
from collections import Counter
from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

# %%
# パスの管理
data_path = Path("data")

df_compare_parmas = [
    {
        "MODEL_NAME": "tohoku-nlp/bert-base-japanese-v3",
        "min_length": 10,
        "use_all_data_for_evaluation": True,
    },
    {
        "MODEL_NAME": "tohoku-nlp/bert-base-japanese-v3",
        "min_length": 20,
        "use_all_data_for_evaluation": True,
    },
    {
        "MODEL_NAME": "tohoku-nlp/bert-base-japanese-v3",
        "min_length": 10,
        "use_all_data_for_evaluation": False,
    },
    {
        "MODEL_NAME": "tohoku-nlp/bert-base-japanese-v3",
        "min_length": 20,
        "use_all_data_for_evaluation": False,
    },
]

df_compares = []
datas = []
for df_compare_param in df_compare_parmas:
    with open(
        data_path
        / f"df_compare_{df_compare_param['MODEL_NAME'].split('/')[-1]}_TextMinLength{df_compare_param['min_length']}_{'All' if df_compare_param['use_all_data_for_evaluation'] else 'Test'}Data.pickle",
        "rb",
    ) as f:
        df_compare = pickle.load(f)
    df_compares.append(df_compare)
    datas.append(
        pd.read_csv(
            data_path / f"data_long_texts_{df_compare_param['min_length']}.tsv",
            sep="\t",
        )
    )

# %%
df_compares[0]

# %%
for i in range(len(df_compares)):

    df_compare = df_compares[i]

    # ラベルのエンコード
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(datas[i]["ラベル"])
    num_labels = len(label_encoder.classes_)

    # ラベルと予測ラベルのリストを取得
    labels = df_compare["ラベル"].tolist()
    predicted_labels = df_compare["ラベル予測"].tolist()

    # ラベルの分布を確認
    label_counter = Counter(labels)

    # 各ラベルのF1スコアを計算
    precision, recall, f1_label, _ = precision_recall_fscore_support(
        labels, predicted_labels, average=None, labels=list(set(labels))
    )

    # 結果をDataFrameにまとめる
    result = pd.DataFrame(
        {
            "label": list(set(labels)),
            "count": [label_counter[i] for i in set(labels)],
            "Precision": precision,
            "Recall": recall,
            "f1_label": f1_label,
        }
    )
    # カテゴリについてのF1スコアを表示
    macro_f1_label = f1_label.mean()
    weighted_f1_label = (f1_label * result["count"]).sum() / result["count"].sum()
    print(f"min_length: {df_compare_parmas[i]['min_length']}")
    print(f"data count: {len(df_compare)}")
    print(f"macro_f1_label: {macro_f1_label}")
    print(f"weighted_f1_label: {weighted_f1_label}")

    # 満足度についてのF1スコアを表示
    print(
        f"f1_satisfaction: {f1_score(df_compare['満足度'], df_compare['満足度予測'], average='macro')}"
    )
    print()

# %%
# 満足度のf1スコアを計算

f1_satisfaction = f1_score(
    df_compare["satisfaction"],
    df_compare["satisfaction_prediction"],
    average="weighted",
)

print(f"weighted-f1(満足度): {f1_satisfaction:.2%}")

# %%
macro_f1_label = f1_label.mean()
weighted_f1_label = (f1_label * result["count"]).sum() / result["count"].sum()
print(f"macro_f1_label: {macro_f1_label}")
print(f"weighted_f1_label: {weighted_f1_label}")
