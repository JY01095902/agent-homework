# 智能文档问答 Agent：Python + Tesseract（OCR）+ FastAPI
# 构建：docker build -t agent-homework .
# 运行：docker run --rm -p 8000:8000 -e OPENAI_API_KEY agent-homework

# 使用官方常用 tag，避免部分镜像站未同步 bookworm 子标签导致拉取失败
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PERSIST_DIR=/app/data/chroma_db \
    HOST=0.0.0.0 \
    PORT=8000

# pytesseract 依赖系统 tesseract；中文手册需 chi_sim + eng
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY data ./data

# 与本地一致：在 src 目录下启动，保证 `import rag` 等相对导入可用
WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
