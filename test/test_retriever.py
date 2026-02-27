from rag.retriever import Retriever

retriever = Retriever()

query = "What is the notice period?"
context = retriever.get_context(query)

print("Retrieved Context:\n")
print(context)