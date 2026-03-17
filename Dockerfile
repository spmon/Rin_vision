FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Gom tất cả lệnh RUN vào làm một và xóa cache ngay lập tức
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Cài xong xóa luôn cache của pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765 
CMD ["python3", "server.py"]