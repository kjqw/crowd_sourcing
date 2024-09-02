# %%
import ast
import json
import os
import pickle
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# %%
# トークン数を記録する変数を初期化
total_input_tokens = 0
total_output_tokens = 0

# %%
# 使用するモデルの設定
MODEL_NAME = "gpt-3.5-turbo"

# 文章データの最小文字数
# min_length = 10
min_length = 20

# %%
# .envファイルのロード
load_dotenv(dotenv_path=Path("data/.env"))

# OpenAI APIキーの設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# 元データのロード
df_data = pd.read_csv(
    f"/workspace/crowd_sourcing/demo/bert/data/data_long_texts_{min_length}.tsv",
    sep="\t",
)
# カラム名の変更
df_data.columns = ["label", "satisfaction", "text", "ID"]
df_data = df_data[["label", "satisfaction", "text"]]

# バッチデータのロード
with open(f"data/df_batches_{min_length}.pickle", "rb") as f:
    df_batches = pickle.load(f)

# モデルのコスト情報のロード
with open("model_costs.json", "r") as f:
    model_costs = json.load(f)

# ラベルのロード
with open("labels.json", "r") as f:
    labels = json.load(f)["labels"]


# %%
def api_request(texts_dict: dict[str], labels: list[str], model: str) -> dict:
    """
    文章を指定されたラベルに分類し、満足か不満かを判断する関数

    Parameters
    ----------
    texts_dict : dict[str]
        分類する文章の辞書
    labels : list[str]
        分類するラベル
    model : str
        使用するモデル

    Returns
    -------
    dict
        分類結果と満足度の判定結果
    """

    global total_input_tokens, total_output_tokens

    try:
        # OpenAI APIリクエストの作成
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {
                #     "role": "system",
                #     "content": "あなたは、テキストを事前定義されたラベルに分類し、満足度を評価するアシスタントです。",
                # },
                {
                    "role": "user",
                    "content": f"与えられた文章を次のようなラベルの1つに分類してください: {labels}。"
                    + "\n次に、その文章が満足か不満のどちらであるかを判断してください。\n出力の形式は次のような辞書型にしてください: {text_id : {'label_prediction': 'ラベル名', 'satisfaction_prediction': '満足' または '不満'}, text_id: ...}。"
                    + f"\n文章: {texts_dict}",
                },
            ],
            temperature=1.0,
            # max_tokens=100,
        )

        # レスポンスからトークン数を取得
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        # 結果の抽出
        result_text = response.choices[0].message.content
        # 文字列を辞書型に変換
        result_dict = ast.literal_eval(result_text)

        return {
            "result_dict": result_dict,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response,
        }

    except Exception as e:
        return {"error": str(e)}


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    使用したトークン数に基づいてコストを計算する関数

    Parameters
    ----------
    input_tokens : int
        入力トークン数
    output_tokens : int
        出力トークン数
    model_name : str
        使用したモデルの名前

    Returns
    -------
    float
        トークン数に基づいて計算されたコスト(単位: USD)
    """

    input_cost = input_tokens / 1e6 * model_costs[model_name]["input"]
    output_cost = output_tokens / 1e6 * model_costs[model_name]["output"]
    return input_cost + output_cost


# %%
result_dict = {}
response_log = []

for i, df_batch in enumerate(df_batches):
    # インデックスをキー、文章を値とする辞書に変換
    texts_dict = df_batch["文章"].to_dict()

    # 分類と評価を実行
    response = api_request(texts_dict, labels, MODEL_NAME)

    # 応答を記録
    response_log.append(response)
    result_dict.update(response["result_dict"])

    # 途中経過を表示
    print(f"Batch {i + 1} / {len(df_batches)}")

# %%
# ログを保存
with open(f"data/response_log_{min_length}.pickle", "wb") as f:
    pickle.dump(response_log, f)

# %%
# 結果をデータフレームに変換
df_result = pd.DataFrame(result_dict).T

# %%
# こちらが要求する形式になっていないものを抽出
df_invalid = df_result[
    (
        (df_result["satisfaction_prediction"] != "満足")
        & (df_result["satisfaction_prediction"] != "不満")
    )  # 満足度の予測が指定したものでないもの
    | ~(
        df_result["label_prediction"].isin(labels)
    )  # ラベルの予測が指定したものでないもの
]

df_invalid

# %%
# df_invalidをdf_resultから取り除く
df_valid = df_result[~df_result.index.isin(df_invalid.index)]

df_valid

# %%
df_compare = pd.concat([df_valid, df_data], axis=1)  # 元データと結果を結合
df_compare["is_match_label"] = (
    df_compare["label"] == df_compare["label_prediction"]
)  # ラベルの一致を判定
df_compare["is_match_satisfaction"] = (
    df_compare["satisfaction"] == df_compare["satisfaction_prediction"]
)  # 満足度の一致を判定
df_compare = df_compare.dropna()  # 欠損値を取り除く

# %%
df_compare

# %%
# 正解率の計算
accuracy_label = df_compare["is_match_label"].mean()
accuracy_satisfaction = df_compare["is_match_satisfaction"].mean()

# コストの計算
total_cost = calculate_cost(total_input_tokens, total_output_tokens, MODEL_NAME)

# 結果の表示
print(f"正解率(ラベル): {accuracy_label:.2%}")
print(f"正解率(満足度): {accuracy_satisfaction:.2%}")
print(f"コスト: ${total_cost:.2f}")

# %%
with open(f"data/df_compare_{MODEL_NAME}_{min_length}.pickle", "wb") as f:
    pickle.dump(df_compare, f)
