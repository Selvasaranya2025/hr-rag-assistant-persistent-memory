import os
from dotenv import load_dotenv
# Load environment variables from a .env file (like OPENAI_API_KEY) to keep sensitive information out of the codebase
load_dotenv()
# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Configuration constants for the RAG system, including model names and parameters for retrieval and chunking
PINECONE_INDEX_NAME = "hr-policy-index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3
CHUNK_SIZE = 500
#memory
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "hr_assistant"
MONGO_COLLECTION = "conversations"
MAX_HISTORY_MESSAGES = 10
