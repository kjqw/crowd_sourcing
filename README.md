# README

## 必要なライブラリ

- pytorch
- transformers
- openai
- scikit-learn
- pandas
- dotenv

## 使い方

### openai

1. `openai`のAPIキーを取得し、`demo/openai/data/.env`ファイルに記述する

   ```.env
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
   ```

### bert

整備中

## メモ

```sh
sudo apt install -y libtiff5
```

`libtiff5`がないとmambaでインストールできなくなっていた。noarchの部分で途中停止した。

```sh
mamba install -y pandas openai python-dotenv scikit-learn nbstripout
```

```sh
nbstripout --install --attributes .gitattributes
```
