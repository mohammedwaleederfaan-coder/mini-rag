FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag.py ingest.py app.py ./

# Ollama runs on the host machine, not inside the container
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    CHROMA_DIR=/app/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]