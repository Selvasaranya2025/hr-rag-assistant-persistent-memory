from rag.loader import load_and_chunk

chunks = load_and_chunk()
print(len(chunks))
print(chunks[0])