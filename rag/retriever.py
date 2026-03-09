# rag/retriever.py

from pinecone import Pinecone
from rag.embedder import get_embeddings
from config import TOP_K
import os
from dotenv import load_dotenv
# Load environment variables from a .env file (like OPENAI_API_KEY) to keep sensitive
# information out of the codebase
load_dotenv()
# Initialize the Pinecone client and connect to the specified index for retrieval operations
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("hr-policy-index")

# The Retriever class encapsulates the logic for retrieving relevant chunks of text from the Pinecone index based on a user query.
class Retriever:
    def retrieve(self, query, top_k=TOP_K):
        query_embedding = get_embeddings([query])[0]

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        chunks = []
        for match in results["matches"]:
            chunks.append(match["metadata"]["text"])

        return chunks
# The get_context method retrieves the top-k relevant chunks
#  for a given query and combines them into a single string 
# with separators for better readability when passed to the LLM.
    def get_context(self, query, top_k=TOP_K):
        retrieved_chunks = self.retrieve(query, top_k)
        return "\n\n---\n\n".join(retrieved_chunks)