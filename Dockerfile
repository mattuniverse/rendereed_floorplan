FROM python:3.11-slim

# System deps needed by opencv/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# If you are baking the model into the image, also copy it here, e.g.:
# COPY best.pt .

ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
