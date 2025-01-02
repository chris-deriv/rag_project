"""Configuration settings for the RAG application."""
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

# OpenAI API settings
OPENAI_API_KEY = get_env_str("OPENAI_API_KEY")
OPENAI_MODEL = get_env_str("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = get_env_float("OPENAI_TEMPERATURE", 0.3)
OPENAI_MAX_TOKENS = get_env_int("OPENAI_MAX_TOKENS", 1000)

# Document processing settings
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 50)

# Response settings
SYSTEM_PROMPT = get_env_str(
    "SYSTEM_PROMPT",
    "You are a helpful assistant that provides detailed and accurate responses."
)
SOURCE_CITATION_PROMPT = get_env_str(
    "SOURCE_CITATION_PROMPT",
    "You are a helpful assistant that provides detailed responses with source citations."
)

# Cache settings
RESPONSE_CACHE_SIZE = get_env_int("RESPONSE_CACHE_SIZE", 1000)
CACHE_ENABLED = get_env_bool("CACHE_ENABLED", True)

# Database settings
CHROMA_COLLECTION_NAME = get_env_str("CHROMA_COLLECTION_NAME", "documents")
CHROMA_PERSIST_DIR = get_env_str("CHROMA_PERSIST_DIR", "./chroma_db")

# File upload settings
UPLOAD_FOLDER = get_env_str("UPLOAD_FOLDER", "./uploads")
MAX_CONTENT_LENGTH = get_env_int("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)  # 16MB
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

# Embedding settings
EMBEDDING_MODEL_NAME = get_env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Settings dictionaries for dynamic settings
LLM_SETTINGS = {
    'temperature': OPENAI_TEMPERATURE,  # 0.3
    'max_tokens': OPENAI_MAX_TOKENS,    # 1000
    'model': OPENAI_MODEL               # "gpt-3.5-turbo"
}

DOCUMENT_PROCESSING_SETTINGS = {
    'chunk_size': CHUNK_SIZE,       # 500
    'chunk_overlap': CHUNK_OVERLAP  # 50
}

CACHE_SETTINGS = {
    'enabled': CACHE_ENABLED,           # True
    'size': RESPONSE_CACHE_SIZE         # 1000
}
