# README

これはクラウドソーシングで得られたアンケートデータを教師データにして、文章から内容と満足度を予測するモデルを作成するリポジトリです。

## 環境構築

私と同じ環境であるならば、VSCodeのDevContainerを使ってDockerコンテナを立ち上げることで環境構築が完了します。そうでない場合は以下のライブラリがあれば動くと思います。

- pytorch
- transformers
- scikit-learn
- pandas
- matplotlib
- japanize-matplotlib
- ipykernel

### 私の環境

WSLのUbuntuにDockerをインストールしています。Windows環境にDockerはありません。

- Windows 11 Home
  - VSCode
  - WSL2
    - Ubuntu 22.04 LTS
      - NVIDIA Container Toolkit
      - Docker
        - pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
          - ここで開発している
  - CUDA 12.1
  - NVIDIA GeForce RTX 3070

Ubuntu 22.04 LTSの`/home/kjqw/university/.cache/`にtransformersのキャッシュフォルダを作成しています。これを`.devcontainer/docker-compose.yml`の設定で、コンテナ内の`workspace/.cache/`にマウントしています。コンテナを消したり、違うコンテナを起動したときに、Hugging Faceからモデルをダウンロードし直さなくて済むようにするためです。

## 処理手順

1. アンケートデータの前処理 `preprocess.py`
2. モデルの学習 `train.py`
3. 学習済みモデルで推論 `predict.py`
4. 推論結果の評価 `evaluate.py`
5. 推論結果の可視化 `visualize.py`

## データ

アンケート内容: `crowdsourcing/アンケート項目.json`
アンケート結果: `crowdsourcing/20240217_地域幸福度/【神奈川県民限定】地域幸福度についてのアンケート/3589290434_row.tsv`

## ファイルの説明

- `classes.py`: データセットやモデルのクラスを定義
- `functions.py`: データの前処理や評価指標の計算などの関数を定義
- `zeroshot.py`: ゼロショット分類のサンプルコード。未完成
- `demo/`: 改善前のコードたち
