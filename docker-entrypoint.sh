#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/chroma_db
mkdir -p /app/model_cache

# Execute the main command
exec "$@"
