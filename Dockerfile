FROM node:24-slim AS frontend

WORKDIR /frontend

COPY app/web/package*.json ./
RUN npm ci

COPY app/web/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libturbojpeg0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision --index-url ${TORCH_INDEX_URL} \
    && python -m pip install -r requirements.txt

COPY . .
COPY --from=frontend /frontend/dist ./app/web/dist

RUN mkdir -p outputs/uploads outputs/api_results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
