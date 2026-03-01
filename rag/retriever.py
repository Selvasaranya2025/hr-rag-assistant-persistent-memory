# rag/retriever.py

import os
import pickle
import numpy as np
# Retriever class for fetching relevant chunks based on query embeddings
from rag.embedder import get_embeddings
from rag.vector_store import load_index, create_faiss_index, save_index
from rag.loader import load_and_chunk
from config import TOP_K, CHUNK_SIZE
# Constants for storage paths
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# Retriever class to handle retrieval of relevant chunks based on query embeddings
class Retriever:
    def __init__(self):

        # 🔥 Auto-build index if missing (for Streamlit Cloud)
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            print("Index not found. Building index...")

            chunks = load_and_chunk(
                chunk_size=CHUNK_SIZE,
            )

            texts = [chunk["content"] for chunk in chunks]
            embeddings = get_embeddings(texts)

            index = create_faiss_index(embeddings)

            os.makedirs(STORAGE_DIR, exist_ok=True)
            save_index(index, INDEX_PATH)

            with open(CHUNKS_PATH, "wb") as f:
                pickle.dump(chunks, f)

            print("Index built successfully.")

        # Load existing index + chunks
        self.index = load_index(INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
# Retrieve relevant chunks based on query embedding
    def retrieve(self, query, top_k=TOP_K):
        query_embedding = get_embeddings([query])[0]
        query_embedding = np.array([query_embedding]).astype("float32")

        _, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results
# Get combined context from retrieved chunks
    def get_context(self, query, top_k=TOP_K):
        retrieved_chunks = self.retrieve(query, top_k)

        context = "\n\n---\n\n".join(
            chunk["content"] for chunk in retrieved_chunks
        )

        return context