# build_index.py

import os
import pickle
# Allow imports from project root
from rag.loader import load_and_chunk
from rag.embedder import get_embeddings
from rag.vector_store import create_faiss_index, save_index
from config import CHUNK_SIZE
# Paths for saving index and chunks
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# This script loads documents, chunks them, creates embeddings, builds a FAISS index, and saves everything for later retrieval
def main():
    print("Loading and chunking documents...")
    chunks = load_and_chunk(
        chunk_size=CHUNK_SIZE,
    )
    # Extract the text content from the chunks to create embeddings
    texts = [chunk["content"] for chunk in chunks]
    # Create embeddings for the chunks using the embedding model
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = get_embeddings(texts)
    # Build the FAISS index using the created embeddings
    print("Building FAISS index...")
    index = create_faiss_index(embeddings)
    # Save the FAISS index and the chunks metadata for later retrieval
    print("Saving index and chunks...")
    save_index(index, INDEX_PATH)
    # Save the chunks metadata (like content) separately since FAISS only stores vectors
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    # Print a success message after the index is built and saved successfully
    print("Index built and saved successfully.")

# Run the script to build the index
if __name__ == "__main__":
    main()