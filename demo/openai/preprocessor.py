# %%
import pickle
from pathlib import Path

import pandas as pd

# %%
# 学習データの最小文字数
# min_length = 10
min_length = 20

# %%
# データの読み込み
input_data_path = (
    Path("/workspace/crowd_sourcing/demo/bert/data")
    / f"data_long_texts_{min_length}.tsv"
)
df = pd.read_csv(input_data_path, sep="\t")
# df

# %%
# バッチサイズごとにデータを分割して保存する

batch_size = 64  # バッチサイズを設定

# バッチサイズごとにデータを分割
df_batches = [df[i : i + batch_size] for i in range(0, len(df), batch_size)]

# %%
print(len(df_batches))
df_batches[0]

# %%
# 保存
with open(f"data/df_batches_{min_length}.pickle", "wb") as f:
    pickle.dump(df_batches, f)
