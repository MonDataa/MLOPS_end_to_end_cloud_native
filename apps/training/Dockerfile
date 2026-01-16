FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
COPY train.py .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]
