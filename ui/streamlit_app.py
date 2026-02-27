# ui/streamlit_app.py
import streamlit as st
import sys
import os
# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.openai_client import stream_chat_completion
from rag.retriever import Retriever
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="HR Policy RAG Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 HR Policy AI Assistant")
st.caption("Ask questions about company HR policies")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙ Settings")

show_context = st.sidebar.toggle(
    "Show Retrieved Context (Debug Mode)",
    value=False
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This AI assistant uses Retrieval-Augmented Generation (RAG).

    **Stack:**
    - FAISS Vector DB
    - OpenAI Embeddings
    - GPT Chat Model
    - Streamlit UI
    """
)


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

retriever = Retriever()


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Quick Question Buttons
# --------------------------------------------------
st.markdown("### 💡 Quick Questions")

col1, col2, col3 = st.columns(3)

if col1.button("Annual Leave Policy"):
    st.session_state.quick_question = "Explain the annual leave policy."

if col2.button("Notice Period"):
    st.session_state.quick_question = "What is the notice period?"

if col3.button("Insurance Coverage"):
    st.session_state.quick_question = "What does health insurance cover?"


# --------------------------------------------------
# Resolve Prompt (Button OR Chat Input)
# --------------------------------------------------
prompt = None

if "quick_question" in st.session_state:
    prompt = st.session_state.quick_question
    del st.session_state.quick_question
else:
    prompt = st.chat_input("Type your HR question here...", key="chat_input_main")


# --------------------------------------------------
# Process User Prompt
# --------------------------------------------------
if prompt:

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    context = retriever.get_context(prompt)

    # Optional Debug Mode
    if show_context:
        with st.expander("🔎 Retrieved Context"):
            st.write(context)

    # Generate streaming response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for token in stream_chat_completion(prompt, context):
            full_response += token
            response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )