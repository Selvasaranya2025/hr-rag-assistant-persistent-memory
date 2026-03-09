# build_index.py
from pinecone import Pinecone
from rag.embedder import get_embeddings
from rag.loader import load_and_chunk

import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("hr-policy-index")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "COMPANY HR POLICY MANUAL.txt")

print("Resolved file path:", file_path)
print("Exists:", os.path.exists(file_path))

chunks = load_and_chunk("data")

print("Chunks length:", len(chunks))

if len(chunks) == 0:
    raise ValueError("No chunks created. Loader failed.")

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i} keys:", chunk.keys())
    print(f"Chunk {i} content type:", type(chunk.get("content")))
    print(f"Chunk {i} content preview:", str(chunk.get("content"))[:100])
    print("-" * 50)

texts = [chunk.get("content") for chunk in chunks]

print("First text type:", type(texts[0]))

embeddings = get_embeddings(texts)
print("First embedding type:", type(embeddings[0]))
print("First embedding length:", len(embeddings[0]))
print("First embedding preview:", embeddings[0][:5])

vectors = []

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {
            "text": chunk["content"]
        }
    })

print("Upserting vectors...")
index.upsert(vectors=vectors)
print("Upserted", len(vectors), "vectors successfully.")