#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/chroma_db
mkdir -p /app/model_cache
mkdir -p /app/uploads
mkdir -p /app/tmp
mkdir -p /app/tmp/libreoffice
mkdir -p /home/appuser/.config/libreoffice/4/user

# Set proper permissions
chown -R appuser:appuser /app/tmp
chmod -R 777 /app/tmp
chmod -R 777 /app/tmp/libreoffice

# Execute the main command
exec "$@"
