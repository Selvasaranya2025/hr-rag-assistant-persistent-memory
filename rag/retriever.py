# rag/retriever.py
import os
import pickle
import numpy as np
# Allow imports from project root
from rag.embedder import get_embeddings
from rag.vector_store import load_index
from config import TOP_K
# Paths for loading index and chunks
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# Retriever class to handle retrieval of relevant chunks based on user query
class Retriever:
    def __init__(self):
        self.index = load_index(INDEX_PATH)
# Load chunks metadata (like content) separately since FAISS only stores vectors
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
# Retrieve top-k relevant chunks for a given query
    def retrieve(self, query, top_k=TOP_K):
        query_embedding = get_embeddings([query])[0]
        query_embedding = np.array([query_embedding]).astype("float32")
# Perform similarity search in the FAISS index
        _, indices = self.index.search(query_embedding, top_k)
# Retrieve the corresponding chunks based on the indices returned by FAISS
        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])
# Return the list of retrieved chunks with their content and source metadata
        return results
# Get the combined context from the retrieved chunks to be used in the prompt for the LLM
    def get_context(self, query, top_k=TOP_K):
        retrieved_chunks = self.retrieve(query, top_k)
# Combine the content of the retrieved chunks into a single context string
        context = "\n\n---\n\n".join(
            chunk["content"] for chunk in retrieved_chunks
        )

        return context
    # For debugging purposes, we can also return the sources of the retrieved chunks