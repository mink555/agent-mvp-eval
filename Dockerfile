FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

ENV EMBEDDING_MODEL=intfloat/multilingual-e5-large
RUN python scripts/init_vectordb.py

ENV API_PORT=7860
EXPOSE 7860

CMD ["python", "run.py"]
