# rag/embedder.py

from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(texts, model=EMBEDDING_MODEL):
    response = client.embeddings.create(
        model=model,
        input=texts
    )

    return [item.embedding for item in response.data]