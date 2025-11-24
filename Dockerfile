FROM python:3.12-slim

WORKDIR /app

# System deps (optional minimal)
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY models ./models
COPY data ./data
COPY retrain.py ./
COPY client_test.py ./
COPY README.md ./

# Expose FastAPI default port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
