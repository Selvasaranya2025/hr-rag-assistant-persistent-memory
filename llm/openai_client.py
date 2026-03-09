# llm/openai_client.py
# This module contains the OpenAI client setup and a function to stream chat completions from the OpenAI API.
import os
from openai import OpenAI
from config import CHAT_MODEL
import streamlit as st
# Initialize the OpenAI client with the API key from the configuration
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)
# Function to stream chat completions from the OpenAI API based on a user query and retrieved context
def stream_chat_completion(query, context: str, history: str = ""):
    system_prompt = """You are an HR policy assistant.
Answer the user's question strictly based on the provided context.
If the answer is not found in the context, say:
'I cannot find this information in the HR policy document.'"""
#history first, then rag context, then user query. 
# This way the LLM can use the history to understand the conversation flow and the context to find the answer, 
# before seeing the new question.

# Combine the system prompt, retrieved context, and user query into a single prompt for the LLM
    user_prompt = f"""
{history}
HR Policy Document:
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