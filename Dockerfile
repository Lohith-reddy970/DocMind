FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCMIND_OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "docmind.api:app", "--host", "0.0.0.0", "--port", "8000"]
