import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Vector Database Settings
CHROMA_COLLECTION_NAME = "documentation"
CHROMA_PERSIST_DIR = "./chroma_db"  # Directory for persistence

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
