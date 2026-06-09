# OpenAI API key (set this if using OpenAI)
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model provider: 'huggingface' or 'openai'
MODEL_PROVIDER = "openai"  # Change to 'openai' to use OpenAI
FASTAPI_URL = " http://localhost:8000"  # URL for FastAPI backend