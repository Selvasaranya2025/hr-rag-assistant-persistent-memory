"""This module contains functions to load documents from a specified directory,
chunk the text into smaller pieces, and prepare the data for retrieval in a RAG system."""
from pathlib import Path
def load_documents(data_path="data"):
    documents = []
# Load all text files from the specified data directory and read their content into a list of documents with source metadata
    for file in Path(data_path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({
                "source": file.name,
                "content": text
            })

    return documents
# This function takes a long text and splits it into smaller chunks based on a specified chunk size, 
# ensuring that chunks are created at paragraph boundaries for better coherence.
def chunk_text(text, chunk_size=500):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
# Iterate through the paragraphs and build chunks until the chunk size limit is reached, then start a new chunk
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
# Add any remaining text as the last chunk after the loop is done
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
# This function combines the loading and chunking steps to prepare the data for embedding and indexing in the RAG system.
# It loads documents from the specified directory, chunks their content, and returns a list of chunks with source metadata.
def load_and_chunk(data_path="data", chunk_size=500):
    documents = load_documents(data_path)
    all_chunks = []
# For each document, chunk the text and create a list of chunks with their corresponding source information
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size)
        for chunk in chunks:
            all_chunks.append({
                "source": doc["source"],
                "content": chunk
            })

    return all_chunks