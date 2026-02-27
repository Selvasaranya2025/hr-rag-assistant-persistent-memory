# llm/openai_client.py
# This module contains the OpenAI client setup and a function to stream chat completions from the OpenAI API.
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL
# Initialize the OpenAI client with the API key from the configuration
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to stream chat completions from the OpenAI API based on a user query and retrieved context
def stream_chat_completion(query, context):
    system_prompt = """You are an HR policy assistant.
Answer the user's question strictly based on the provided context.
If the answer is not found in the context, say:
'I cannot find this information in the HR policy document.'"""
# Combine the system prompt, retrieved context, and user query into a single prompt for the LLM
    user_prompt = f"""
Context:
{context}

Question:
{query}
"""
# Stream the chat completion response from the OpenAI API and yield the content as it arrives
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
# As the response is streamed from the OpenAI API, yield the content of each chunk to be displayed in the UI in real-time
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content