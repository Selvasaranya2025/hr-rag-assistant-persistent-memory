from pymongo import MongoClient
from datetime import datetime
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION, MAX_HISTORY_MESSAGES

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
conversations = db[MONGO_COLLECTION]


def save_message(session_id: str, role: str, content: str):
    """Save a single message to MongoDB."""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    conversations.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": message},
            "$set":  {"last_updated": datetime.utcnow().isoformat()}
        },
        upsert=True   # creates doc if session doesn't exist yet
    )


def load_history(session_id: str) -> list:
    """Load full conversation history for a session."""
    doc = conversations.find_one({"session_id": session_id})
    if doc:
        return doc.get("messages", [])
    return []


def get_recent_history(session_id: str) -> list:
    """Return only the last N messages to control token cost."""
    history = load_history(session_id)
    return history[-MAX_HISTORY_MESSAGES:]


def clear_session(session_id: str):
    """Delete all messages for a session."""
    conversations.delete_one({"session_id": session_id})


def format_history_for_llm(history: list) -> str:
    """Convert history list into a readable string for the LLM prompt."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)