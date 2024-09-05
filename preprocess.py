from pathlib import Path

import functions

# フィルタリングする文章の最小の長さを指定
min_length = 10

# データ保存用のディレクトリを指定
data_path = Path(__file__).parent / "data"

# アンケートの生データを読み込む
input_tsv_path = (
    Path(__file__).parent
    / "crowdsourcing/20240217_地域幸福度/【神奈川県民限定】地域幸福度についてのアンケート/3589290434_row.tsv"
)

# アンケートデータの欠損値を削除し、TSVファイルとして保存
output_tsv_path = data_path / "data_without_nan.tsv"
functions.preprocess_tsv(input_tsv_path, output_tsv_path)

# 指定された長さ以上の文章を抽出し、TSVファイルとして保存
output_long_text_path = data_path / f"data_long_texts_{min_length}.tsv"
functions.extract_long_text(output_tsv_path, output_long_text_path, min_length)
