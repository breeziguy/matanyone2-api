FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install SAM2
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/sam2.git" \
    opencv-python-headless \
    rembg \
    onnxruntime \
    pillow \
    numpy \
    runpod

# Download SAM2 weights at build time
RUN mkdir -p /app/weights
COPY download_weights.py .
RUN python download_weights.py

COPY handler.py .

CMD ["python", "handler.py"]
