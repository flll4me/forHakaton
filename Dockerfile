FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/models/hf
WORKDIR /app

# CPU-сборка torch, чтобы не тянуть CUDA
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY Makefile ./

# Данные кейса и обученная модель в образ не кладутся (их нельзя публиковать) — монтируются томами:
#   docker run -p 8000:8000 -v $PWD/data:/app/data -v $PWD/models:/app/models gm-copilot
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
