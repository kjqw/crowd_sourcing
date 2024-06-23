# README

整備中

## 必要なライブラリ

- pandas
- matplotlib
- japanize-matplotlib
- scikit-learn
- dotenv
- pytorch
- transformers
- openai

## 使い方

### bert

1. `demo/bert/preprocessor.ipynb`を実行
2. `demo/bert/train.ipynb`を実行
3. `demo/bert/evaluate.ipynb`を実行

### openai

1. `openai`のAPIキーを取得し、`demo/openai/data/.env`ファイルに記述する

   ```.env
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
   ```

## メモ

### エラー対処

```sh
sudp apt update && sudo apt install -y libtiff5
```

`libtiff5`がないとmambaでインストールできなくなっていた。`mamba install ...`の際、noarchの部分で途中停止した。
Pythonで`transformers`をインポートする際にもエラーが出た。

### nbstripout

```sh
mamba install -y nbstripout
```

```sh
nbstripout --install --attributes .gitattributes
```
