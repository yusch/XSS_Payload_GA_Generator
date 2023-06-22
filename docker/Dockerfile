FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update \
    && apt install -y \
    libgl1-mesa-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libjpeg-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    wget \
    tidy \
    neovim \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# cuda path
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
# ENV CUDA_HOME=/usr/local/cuda

# Install Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
    && tar -xvf Python-3.11.4.tgz \
    && cd Python-3.11.4 \
    && ./configure --enable-optimizations --enable-shared \
    && make altinstall \
    && ldconfig \
    && rm -rf Python-3.11.4.tgz /opt/Python-3.11.4 \
    && ln -sf /usr/local/bin/python3.11 /opt/conda/bin/python3 \
    && ln -sf /usr/local/bin/python3.11 /opt/conda/bin/python

COPY requirements.txt .
RUN python3 -m pip --no-cache-dir install -U pip==23.1.2
RUN python3 -m pip --no-cache-dir install -r requirements.txt
RUN pip install torch
RUN pip install torchvision