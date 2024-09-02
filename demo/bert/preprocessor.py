# %%
from pathlib import Path

import pandas as pd

# %%
data_path = Path.cwd() / "data"

# データ保存用のディレクトリを作成
data_path.mkdir(exist_ok=True)


# %%
def preprocess_tsv(input_file: str, output_file: str) -> None:
    """
    指定されたTSVファイルを分類タスクの教師データに変換します。

    Parameters
    ----------
    input_file : str
        入力TSVファイルのパス
    output_file : str
        出力TSVファイルのパス
    """

    # TSVファイルを読み込み
    df = pd.read_csv(input_file, delimiter="\t")

    # 新しいデータフレームを作成
    processed_data = []

    for index, row in df.iterrows():
        # 設問2と設問3 -> ラベル, 満足, 文章, 会員ID
        processed_data.append([row["設問2"], "満足", row["設問3"], row["会員ID"]])
        # 設問4と設問5 -> ラベル, 満足, 文章, 会員ID
        processed_data.append([row["設問4"], "満足", row["設問5"], row["会員ID"]])
        # 設問6と設問7 -> ラベル, 不満, 文章, 会員ID
        processed_data.append([row["設問6"], "不満", row["設問7"], row["会員ID"]])
        # 設問8と設問9 -> ラベル, 不満, 文章, 会員ID
        processed_data.append([row["設問8"], "不満", row["設問9"], row["会員ID"]])

    # データフレームに変換
    processed_df = pd.DataFrame(
        processed_data, columns=["ラベル", "満足度", "文章", "会員ID"]
    )

    # 欠損値の削除
    processed_df.dropna(inplace=True)

    # TSVファイルとして保存
    processed_df.to_csv(output_file, sep="\t", index=False)


# %%
input_tsv_dir = "/workspace/crowd_sourcing/crowdsourcing/20240217_地域幸福度/【神奈川県民限定】地域幸福度についてのアンケート/3589290434_row.tsv"  # 入力のtsvファイルのパスを指定
output_tsv_path = data_path / "data_without_nan.tsv"  # 出力のtsvファイルのパスを指定
preprocess_tsv(input_tsv_dir, output_tsv_path.__str__())  # データの前処理を実行


# %%
def extract_long_text(input_file: str, output_file: str, min_length: int) -> None:
    """
    指定されたTSVファイルから特定の長さ以上の文章を抽出します。

    Parameters
    ----------
    input_file : str
        入力TSVファイルのパス
    output_file : str
        出力TSVファイルのパス
    min_length : int
        抽出する文章の最小の長さ
    """

    # TSVファイルを読み込み
    df = pd.read_csv(input_file, delimiter="\t")

    # 文字数が最小値以上の文章を抽出
    long_texts = df[df["文章"].str.len() >= min_length]

    # TSVファイルとして保存
    long_texts.to_csv(output_file, sep="\t", index=False)


# %%
min_length = 10  # 抽出する文章の最小の長さを指定
input_tsv_path = data_path / "data_without_nan.tsv"  # 入力のtsvファイルのパスを指定
output_tsv_path = (
    data_path / f"data_long_texts_{min_length}.tsv"
)  # 出力のtsvファイルのパスを指定
extract_long_text(
    input_tsv_path.__str__(), output_tsv_path.__str__(), min_length
)  # 長い文章の抽出を実行
