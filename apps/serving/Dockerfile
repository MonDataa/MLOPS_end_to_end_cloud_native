FROM python:3.10-slim

WORKDIR /app

# Install serving dependencies.
COPY apps/serving/requirements.txt requirements.txt
COPY apps/serving/app.py app.py

# Include the Feast feature repo so the feature store config is available
# even if the shared host volume is missing or empty.
COPY feast/feature_repo ./feast/feature_repo

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
