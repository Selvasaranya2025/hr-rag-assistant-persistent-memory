import streamlit as st
import sys, os, uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.openai_client import stream_chat_completion
from rag.retriever import Retriever
from memory.mongo_memory import (
    save_message, load_history, get_recent_history,
    clear_session, format_history_for_llm
)

st.set_page_config(page_title="HR Policy RAG Assistant", page_icon="🏢", layout="wide")
st.title("🏢 HR Policy AI Assistant")
st.caption("Ask questions about company HR policies")

# --- Generate unique session ID per browser session ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# --- Sidebar ---
st.sidebar.title("⚙ Settings")
show_context = st.sidebar.toggle("Show Retrieved Context (Debug Mode)", value=False)
st.sidebar.markdown(f"`Session: {session_id[:8]}...`")

if st.sidebar.button("🧹 Clear Chat"):
    clear_session(session_id)          # wipe MongoDB
    st.session_state.messages = []     # wipe display
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**Stack:**
- Pinecone (Vector DB)
- OpenAI Embeddings + GPT
- MongoDB (Persistent Memory)
- Streamlit UI
""")

# --- On first load, restore history from MongoDB ---
if "messages" not in st.session_state:
    db_history = load_history(session_id)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in db_history
    ]

retriever = Retriever()

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Quick question buttons ---
st.markdown("### 💡 Quick Questions")
col1, col2, col3 = st.columns(3)
if col1.button("Annual Leave Policy"):
    st.session_state.quick_question = "Explain the annual leave policy."
if col2.button("Notice Period"):
    st.session_state.quick_question = "What is the notice period?"
if col3.button("Insurance Coverage"):
    st.session_state.quick_question = "What does health insurance cover?"

# --- Resolve prompt ---
prompt = None
if "quick_question" in st.session_state:
    prompt = st.session_state.quick_question
    del st.session_state.quick_question
else:
    prompt = st.chat_input("Type your HR question here...")

# --- Process prompt ---
if prompt:

    # 1. Save user message to MongoDB immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(session_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Retrieve RAG context from Pinecone
    context = retriever.get_context(prompt)

    # 3. Load recent conversation history from MongoDB
    recent_history = get_recent_history(session_id)
    history_str = format_history_for_llm(recent_history)

    # Debug mode — shows both RAG context and memory
    if show_context:
        with st.expander("🔎 Retrieved Context (Pinecone)"):
            st.write(context)
        with st.expander("🧠 Conversation Memory (sent to LLM)"):
            st.write(history_str if history_str else "No history yet.")

    # 4. Stream response — pass history alongside RAG context
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for token in stream_chat_completion(prompt, context, history_str):
            full_response += token
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)

    # 5. Save assistant response to MongoDB
    save_message(session_id, "assistant", full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})