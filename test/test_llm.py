from rag.retriever import Retriever
from llm.openai_client import stream_chat_completion

retriever = Retriever()

query = "What is the notice period?"
context = retriever.get_context(query)

print("Answer:\n")

for token in stream_chat_completion(query, context):
    print(token, end="", flush=True)