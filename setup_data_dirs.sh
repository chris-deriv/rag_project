#!/bin/bash

# Create data directory structure
mkdir -p data/{chroma_db,model_cache,huggingface_cache,tiktoken_cache,torch_cache}

# Set permissions for current user
chmod -R 755 data

echo "Data directories created and permissions set"
ls -la data/
