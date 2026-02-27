import os
from dotenv import load_dotenv
# Load environment variables from a .env file (like OPENAI_API_KEY) to keep sensitive information out of the codebase
load_dotenv()
# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configuration constants for the RAG system, including model names and parameters for retrieval and chunking
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3
CHUNK_SIZE = 500