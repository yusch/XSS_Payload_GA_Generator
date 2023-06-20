FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    wget \
    tidy

# cuda path
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
ENV CUDA_HOME=/usr/local/cuda

# Install Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
    && tar -xvf Python-3.11.4.tgz \
    && cd Python-3.11.4 \
    && ./configure --enable-optimizations --enable-shared \
    && altinstall \
    && ldconfig \
    && rm -rf Python-3.11.4.tgz /opt/Python-3.11.4 \
    && ln -sf /usr/local/bin/python3.11 /opt/conda/bin/python3 \
    && ln -sf /usr/local/bin/python3.11 /opt/conda/usr/bin/python

COPY requirements.txt .
RUN python3 -m pip install -U --no-cache-dir pip==23.1.2 \
    && pip install --no-cache-dir -r requirements.txt