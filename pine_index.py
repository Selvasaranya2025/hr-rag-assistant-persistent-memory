from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if "hr-policy-index" not in pc.list_indexes().names():
    pc.create_index(
        name="hr-policy-index",
        dimension=1536,  # IMPORTANT: match embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )