# CS_Agent 推理服務（llama-cpp-python + CUDA）
# 從原始碼編譯以取得 CUDA 支援；首次建置約需 10-40 分鐘（視機器）
# base 與桌機 WSL 環境對齊：Ubuntu 26.04 + CUDA 13.4
FROM nvidia/cuda:13.4.3-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_ARGS="-DGGML_CUDA=on"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages fastapi "uvicorn[standard]" \
 && pip3 install --no-cache-dir --break-system-packages llama-cpp-python

WORKDIR /srv
COPY server_llama.py .

EXPOSE 10000
CMD ["uvicorn", "server_llama:app", "--host", "0.0.0.0", "--port", "10000"]
