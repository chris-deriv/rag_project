#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/chroma_db
mkdir -p /app/model_cache
mkdir -p /app/uploads
mkdir -p /tmp/libreoffice
mkdir -p /home/appuser/.config/libreoffice/4/user

# Execute the main command
exec "$@"
