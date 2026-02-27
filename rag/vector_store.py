""" rag/vector_store.py"""
# rag/vector_store.py
import os
import faiss
import numpy as np
# Vector Store using FAISS
def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
# if you want to use a more advanced index (e.g., with IVF, PQ), you can modify this function accordingly
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings) #type: ignore
# if you want to save metadata (like chunk IDs), you would need to maintain a separate mapping since FAISS only stores vectors
    return index

# For simplicity, we'll just save the FAISS index and the chunks separately
def save_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
# To load the index later
def load_index(path):
    return faiss.read_index(path)