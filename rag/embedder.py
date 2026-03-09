# rag/embedder.py

from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)
# Function to get embeddings for a list of texts using the OpenAI API and the specified embedding model
def get_embeddings(texts, model=EMBEDDING_MODEL):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts #sends all texts in one request to the OpenAI API to get their embeddings
          #in a single call, which is more efficient than sending individual requests for each text
    )

    return [item.embedding for item in response.data] 
#response.data contains the list of embedding results, 
# and we extract the embedding vector from each item to return a list of embeddings corresponding to the input texts.
#item.embedding is the actual embedding vector for each input text, 
# which is what we need to store in the vector database for retrieval later on.