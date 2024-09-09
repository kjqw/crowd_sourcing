from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


class CustomDataset(Dataset):
    """
    テキストデータとラベルをBERTに適した形式に変換するカスタムデータセットクラス

    Parameters
    ----------
    texts : list[str]
        入力となる文章のリスト
    labels : list[int]
        各文章に対応するラベルのリスト
    tokenizer : BertTokenizer
        BERT用のトーカナイザ
    max_length : int
        トークン化時の最大長
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: BertTokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """データセット内のサンプル数を返す"""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        指定されたインデックスのデータを返す

        Parameters
        ----------
        idx : int
            データセットのインデックス

        Returns
        -------
        dict[str, torch.Tensor]
            トークン化された入力データとラベルの辞書
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # テキストをトークン化し、BERT用の入力形式に変換
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TextClassifier:
    """
    テキスト分類を行うクラス。BERTを使用してテキスト分類モデルを構築し、訓練や予測、評価を行う。

    Parameters
    ----------
    model_name : str
        使用するBERTモデルの名前
    num_labels : int
        分類するラベルの数
    max_length : int
        トークン化時の最大長
    device : torch.device
        モデルを実行するデバイス (CPU/GPU)
    load_model_path : Path, Optional
        事前学習済みモデルの重みを読み込む場合のパス
    """

    def __init__(
        self,
        model_name: str | Path,
        num_labels: int,
        max_length: int,
        device: torch.device,
        load_model_path: Path | None = None,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if load_model_path:
            self.model = BertForSequenceClassification.from_pretrained(
                load_model_path, num_labels=num_labels
            )
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        self.model = self.model.to(device)
        self.max_length = max_length
        self.device = device

    def train(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> tuple[float, float]:
        """
        モデルの訓練を行う関数

        Parameters
        ----------
        data_loader : DataLoader
            訓練用DataLoader
        optimizer : torch.optim.Optimizer
            モデルの重みを更新するオプティマイザ
        epoch : int
            エポック数

        Returns
        -------
        tuple
            正解率と損失
        """
        model = self.model.train()  # モデルを訓練モードに設定
        losses = []
        correct_predictions = 0

        # データローダーを使ってバッチごとに訓練
        for data in tqdm(data_loader, desc=f"Epoch {epoch}"):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["labels"].to(self.device)

            # モデルにデータを入力し、出力を得る
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            # 最も高い確率のクラスを予測
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)  # 正解数をカウント
            losses.append(loss.item())  # 損失を保存

            # 逆伝播とオプティマイザによる重み更新
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 正解率と平均損失を返す
        return correct_predictions.double() / len(data_loader.dataset), sum(
            losses
        ) / len(losses)

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        モデルの評価を行う関数

        Parameters
        ----------
        data_loader : DataLoader
            評価用DataLoader

        Returns
        -------
        float
            正解率
        """
        model = self.model.eval()  # モデルを評価モードに設定
        correct_predictions = 0

        # 評価モードでは勾配計算を行わない
        with torch.no_grad():
            for data in data_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                # モデルにデータを入力し、予測結果を得る
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

        # 正解率を返す
        return correct_predictions.double() / len(data_loader.dataset)

    def predict_text(self, text: str, label_encoder: LabelEncoder) -> str:
        """
        テキストを分類する関数

        Parameters
        ----------
        text : str
            分類対象のテキスト
        label_encoder : LabelEncoder
            ラベルのエンコーダ

        Returns
        -------
        str
            予測されたラベル
        """
        # テキストをトークン化
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # モデルにデータを入力し、予測結果を得る
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        # ラベルを数値から元の文字列に変換
        return label_encoder.inverse_transform(preds.cpu().numpy())[0]

    def save_model(self, save_path: Path) -> None:
        """
        モデルの重みを指定したパスに保存する関数

        Parameters
        ----------
        save_path : Path
            保存先のパス
        """
        # 保存先ディレクトリが存在しない場合は作成
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
