"""Configuration settings for the RAG application.

This module handles core environment configuration and provides default values for dynamic settings.
It is organized into several categories:

1. Core Environment Settings:
   - Required API keys and essential configuration that must be set in the environment

2. Storage Paths:
   - Directory paths for database, uploads, and other persistent storage
   - These should be configured per deployment environment

3. Model Settings:
   - Configuration for embedding models and other ML components
   - These are typically static per deployment

4. Default Values for Dynamic Settings:
   - Initial values for settings that can be changed at runtime
   - These values are used to initialize the dynamic settings system
   - Changes to these settings should be made through the settings API, not environment variables

For runtime-configurable settings, use the dynamic_settings module instead of modifying
environment variables directly.
"""
import os
from typing import Optional

def get_env_str(key: str, default: Optional[str] = None) -> str:
    """Get environment variable as string."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

def get_env_int(key: str, default: Optional[int] = None) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key, str(default) if default is not None else None)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return int(value)

def get_env_float(key: str, default: Optional[float] = None) -> float:
    """Get environment variable as float."""
    value = os.getenv(key, str(default) if default is not None else None)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return float(value)

def get_env_bool(key: str, default: Optional[bool] = None) -> bool:
    """Get environment variable as boolean."""
    value = os.getenv(key, str(default) if default is not None else None)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value.lower() in ('true', '1', 'yes', 'on')

# Core environment settings
OPENAI_API_KEY = get_env_str("OPENAI_API_KEY")

# Storage paths
CHROMA_COLLECTION_NAME = get_env_str("CHROMA_COLLECTION_NAME", "documents")
CHROMA_PERSIST_DIR = get_env_str("CHROMA_PERSIST_DIR", "./chroma_db")
UPLOAD_FOLDER = get_env_str("UPLOAD_FOLDER", "./uploads")

# Model settings
EMBEDDING_MODEL_NAME = get_env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# File upload settings
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

# Default values for dynamic settings
DEFAULT_MODEL = get_env_str("OPENAI_MODEL", "gpt-4")
DEFAULT_TEMPERATURE = get_env_float("DEFAULT_TEMPERATURE", 0.3)
DEFAULT_MAX_TOKENS = get_env_int("DEFAULT_MAX_TOKENS", 1000)
DEFAULT_CHUNK_SIZE = get_env_int("DEFAULT_CHUNK_SIZE", 500)
DEFAULT_CHUNK_OVERLAP = get_env_int("DEFAULT_CHUNK_OVERLAP", 50)
DEFAULT_CACHE_SIZE = get_env_int("RESPONSE_CACHE_SIZE", 1000)
DEFAULT_CACHE_ENABLED = get_env_bool("RESPONSE_CACHE_ENABLED", True)

# Settings dictionaries for dynamic settings initialization
LLM_SETTINGS = {
    'temperature': DEFAULT_TEMPERATURE,
    'max_tokens': DEFAULT_MAX_TOKENS,
    'model': DEFAULT_MODEL
}

DOCUMENT_PROCESSING_SETTINGS = {
    'chunk_size': DEFAULT_CHUNK_SIZE,
    'chunk_overlap': DEFAULT_CHUNK_OVERLAP
}

CACHE_SETTINGS = {
    'enabled': DEFAULT_CACHE_ENABLED,
    'size': DEFAULT_CACHE_SIZE
}
