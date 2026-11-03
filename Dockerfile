FROM python:3.10-slim

# Install system dependencies for OpenCV and AI inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache requirements installation
COPY requirements.txt requirements-ai.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ai.txt

# Copy source code and pre-trained models
COPY . .

EXPOSE 8001
CMD ["python", "web_app.py"]
