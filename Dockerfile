FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p ./chroma_db

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
