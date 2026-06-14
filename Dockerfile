FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 \
    build-essential cython3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pq-yang/MatAnyone2 /app/MatAnyone2

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -e /app/MatAnyone2

COPY download_weights.py .
RUN python3 download_weights.py

COPY handler.py .

CMD ["python3", "handler.py"]
