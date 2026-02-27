from rag.embedder import get_embeddings

emb = get_embeddings(["Hello world"])
print(len(emb))
print(len(emb[0]))