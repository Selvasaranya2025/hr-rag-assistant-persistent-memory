🏢 HR Policy RAG Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

This system allows users to ask questions about company HR policies and receive answers grounded strictly in official HR documents.

🚀 Features

📄 Document-based Question Answering

🔎 FAISS Vector Search

🧠 OpenAI Embeddings + Chat Model

💬 Streaming responses in Streamlit

🧩 Modular architecture (UI / Retriever / Vector Store / LLM separated)

🗂 Local FAISS index storage

🏗 Architecture
User Question
      ↓
Retriever
      ↓
FAISS Vector Search
      ↓
Relevant HR Chunks
      ↓
OpenAI LLM (with context)
      ↓
Streamlit UI Response

The system follows clean separation of concerns:

ui/ → Frontend (Streamlit)

rag/ → Retrieval logic

llm/ → OpenAI integration

storage/ → FAISS index + chunk metadata

data/ → HR policy document

📁 Project Structure
day9/
│
├── data/
│   └── COMPANY HR POLICY MANUAL.txt
│
├── storage/
│   ├── index.faiss
│   └── chunks.pkl
│
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── retriever.py
│
├── llm/
│   ├── __init__.py
│   └── openai_client.py
│
├── ui/
│   └── streamlit_app.py
│
├── build_index.py
├── config.py
├── requirements.txt
└── README.md
⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone <your-repo-url>
cd day9
2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add Environment Variables

Create a .env file:

OPENAI_API_KEY=your_openai_key_here
5️⃣ Build FAISS Index
python build_index.py

This will:

Load HR document

Chunk text

Generate embeddings

Create FAISS index

Save index + chunks locally

6️⃣ Run Streamlit App
streamlit run ui/streamlit_app.py

Open browser:

http://localhost:8501
🧠 How It Works

HR document is chunked into smaller sections.

Each chunk is converted into vector embeddings.

Embeddings are stored in a FAISS index.

User question is embedded.

Similar chunks are retrieved.

Retrieved context is sent to the LLM.

LLM generates a grounded response.

🔒 Grounded Answering

The assistant is instructed to:

Answer strictly from provided HR context.

Avoid hallucination.

Return fallback message if answer not found.

🛠 Tech Stack

Python 3.12

Streamlit

FAISS (Vector Database)

OpenAI API

NumPy

Pickle

📌 Example Questions

What is the notice period?

How many annual leave days are allowed?

What does health insurance cover?

Is remote work allowed?

🔮 Future Improvements

Pinecone / Cloud Vector DB support

Metadata filtering

Similarity score thresholding

Multi-document ingestion

Authentication layer

Deployment (Streamlit Cloud / Docker)

👩‍💻 Author

Built as part of modular RAG system development practice.