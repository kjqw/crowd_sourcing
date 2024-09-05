# メモ

## 必要なライブラリ

### python

```sh
mamba install -y openai python-dotenv scikit-learn
```

```sh
pip install japanize-matplotlib
```

### nbstripout

```sh
mamba install -y nbstripout
```

```sh
nbstripout --install --attributes .gitattributes
```

## 作業記録

間違ってどこかで400MBの学習後の重みをコミットしてしまったため、それを削除したい。リポジトリのバックアップを取ってから以下を実行した。

```sh
pip install git-filter-repo
```

```sh
git filter-repo --path demo/bert/data/label_model/model.safetensors --path demo/bert/data/satisfaction_model/model.safetensors --invert-paths
```
