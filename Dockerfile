FROM python:3.11-slim

WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    libreoffice-writer-nogui \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Create and set permissions for all directories
RUN mkdir -p \
    /app/chroma_db \
    /app/model_cache \
    /app/cache/huggingface \
    /app/cache/tiktoken \
    /app/cache/torch \
    /app/uploads \
    /app/tmp \
    /home/appuser/.config/libreoffice/4/user \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser/.config \
    && chmod -R 755 /home/appuser/.config \
    && chmod -R 777 /app/tmp \
    && chmod -R 755 /app

# Copy and set up entrypoint script with root permissions
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user before pip install
USER appuser

# Add local bin to PATH for pip --user installs
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Set LibreOffice user profile directory and temp directory
ENV HOME=/home/appuser \
    USER_INSTALLATION='file:///home/appuser/.config/libreoffice/4/user' \
    TMPDIR=/app/tmp

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Set cache environment variables
ENV TRANSFORMERS_CACHE=/app/cache/huggingface \
    TIKTOKEN_CACHE_DIR=/app/cache/tiktoken \
    TORCH_HOME=/app/cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Copy application code
COPY --chown=appuser:appuser . .

EXPOSE 5000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
