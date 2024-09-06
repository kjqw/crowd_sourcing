# %%
from pathlib import Path

import torch
from classes import TextDataset
from functions import evaluate_model, load_data, save_model, train_model
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

# %%
# モデルとデータの設定
MODEL_NAME_ZEROSHOT = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_NAME_BERT = "google-bert/bert-base-multilingual-cased"
DATA_PATH = Path(__file__).parent / "data" / "data_long_texts_10.tsv"  # データのパス
SAVE_DIR = Path(__file__).parent / "model"

# %%
# データ読み込み
data = load_data("./data/data_long_texts_10.tsv")

# %%
# トーカナイザとモデルの設定
model = BertForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=16)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_BERT)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# %%
# データ分割
train_data, test_data = train_test_split(data, test_size=0.2)

# %%
# DataLoaderの設定
train_dataset = TextDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TextDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %%
# モデルの訓練
optimizer = AdamW(model.parameters(), lr=5e-5)
trained_model = train_model(model, train_loader, optimizer)

# %%
# モデルの保存
save_model(trained_model, "./saved_models/model")

# %%
# モデルの評価
accuracy = evaluate_model(trained_model, test_loader)
print(f"Accuracy: {accuracy}")

# %%
