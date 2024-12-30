#!/bin/bash

# This script sets up data directories for local development and testing.
# When running in Docker, these directories are not needed as Docker volumes are used instead.

echo "Creating data directories for local development..."

# Create base data directory
mkdir -p data

# Create subdirectories for different caches
mkdir -p data/chroma_db
mkdir -p data/model_cache
mkdir -p data/huggingface_cache
mkdir -p data/tiktoken_cache
mkdir -p data/torch_cache

# Create uploads directory
mkdir -p uploads

echo "Data directories created successfully."
echo "Note: These directories are only needed for local development."
echo "When running with Docker, managed volumes will be used instead."
