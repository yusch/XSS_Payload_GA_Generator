FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM selenium/standalone-firefox:latest

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    wget \
    tidy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt