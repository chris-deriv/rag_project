FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libreoffice \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create necessary directories and set permissions
RUN mkdir -p \
    /app/chroma_db \
    /app/model_cache \
    /app/cache/huggingface \
    /app/cache/tiktoken \
    /app/cache/torch \
    && chown -R appuser:appuser /app

# Set cache environment variables
ENV TRANSFORMERS_CACHE=/app/cache/huggingface \
    TIKTOKEN_CACHE_DIR=/app/cache/tiktoken \
    TORCH_HOME=/app/cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Copy application code
COPY . .

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set ownership of all files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 5000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
