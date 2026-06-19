import os
import json
import uuid
from datetime import datetime

CONVERSATIONS_DIR = os.path.join(os.getcwd(), "conversations")


def _ensure_dir():
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


def new_conversation() -> dict:
    """Create and return a new empty conversation dict (not yet saved to disk)."""
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }


def save(conv: dict):
    """Persist a conversation dict to disk as JSON."""
    _ensure_dir()
    conv["updated_at"] = datetime.now().isoformat()
    path = os.path.join(CONVERSATIONS_DIR, f"{conv['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


def load(conv_id: str) -> dict | None:
    """Load a single conversation by ID. Returns None if not found."""
    path = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_all() -> list[dict]:
    """Load all saved conversations, sorted newest-first by updated_at."""
    _ensure_dir()
    convs = []
    for fname in os.listdir(CONVERSATIONS_DIR):
        if fname.endswith(".json"):
            conv = load(fname[:-5])
            if conv:
                convs.append(conv)
    convs.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return convs


def delete(conv_id: str):
    """Delete a conversation file from disk."""
    path = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        os.remove(path)
