FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY apps/training/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/training/train.py train.py
COPY apps/__init__.py ./apps/__init__.py
COPY apps/monitoring ./apps/monitoring

CMD ["python", "train.py"]
