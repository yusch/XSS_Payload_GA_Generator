# ベースイメージの指定
FROM python:3.11.2

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    wget

# ワーキングディレクトリの設定
WORKDIR /app

# 必要なPythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tidyのインストール
RUN apt-get install -y tidy

# コンテナの起動時に実行するコマンド
CMD ["bash"]