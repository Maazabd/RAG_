"""
conversation_manager.py
-----------------------
Stores chat conversations in Supabase (JSONB column for messages).
Falls back to local JSON files if Supabase is not configured.

Supabase table schema (run once in the Supabase SQL editor):

    CREATE TABLE conversations (
        id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        title       TEXT        NOT NULL DEFAULT 'New Chat',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        messages    JSONB       NOT NULL DEFAULT '[]'::jsonb
    );

    -- Optional: index for fast recency sorting
    CREATE INDEX idx_conversations_updated ON conversations (updated_at DESC);
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional

# ── Local fallback directory ──────────────────────────────────────────
_LOCAL_DIR = os.path.join(os.getcwd(), "conversations")
_TABLE     = "conversations"

# ── Supabase client (lazy, singleton) ────────────────────────────────
_client = None


def _get_client():
    """Lazy-initialize and return the Supabase client.

    Reads credentials from st.secrets (Streamlit Cloud / local secrets.toml)
    then falls back to environment variables.
    Returns None if no credentials are found.
    """
    global _client
    if _client is not None:
        return _client

    url = key = ""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        pass

    url = url or os.getenv("SUPABASE_URL", "")
    key = key or os.getenv("SUPABASE_KEY", "")

    if url and key:
        try:
            from supabase import create_client
            _client = create_client(url, key)
            print("✅ Supabase client initialized")
        except Exception as e:
            print(f"⚠️  Supabase init failed, using local fallback: {e}")

    return _client


# ── Public API ────────────────────────────────────────────────────────

def new_conversation() -> dict:
    """Create and return a new empty conversation dict (not yet saved to DB)."""
    return {
        "id":         str(uuid.uuid4()),
        "title":      "New Chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages":   [],
    }


def save(conv: dict):
    """Upsert a conversation.  Tries Supabase first, falls back to local files."""
    conv["updated_at"] = datetime.now().isoformat()

    client = _get_client()
    if client:
        try:
            row = {
                "id":         conv["id"],
                "title":      conv.get("title", "New Chat"),
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"],
                "messages":   conv.get("messages", []),   # Supabase stores JSONB natively
            }
            client.table(_TABLE).upsert(row).execute()
            return
        except Exception as e:
            print(f"Supabase save error (falling back to file): {e}")

    _file_save(conv)


def load(conv_id: str) -> Optional[dict]:
    """Load a single conversation by ID.  Returns None if not found."""
    client = _get_client()
    if client:
        try:
            resp = (
                client.table(_TABLE)
                .select("*")
                .eq("id", conv_id)
                .maybe_single()
                .execute()
            )
            if resp.data:
                return _normalise(resp.data)
        except Exception as e:
            print(f"Supabase load error (falling back to file): {e}")

    return _file_load(conv_id)


def load_all() -> list[dict]:
    """Return all conversations sorted newest-first by updated_at."""
    client = _get_client()
    if client:
        try:
            resp = (
                client.table(_TABLE)
                .select("*")
                .order("updated_at", desc=True)
                .execute()
            )
            if resp.data is not None:
                return [_normalise(r) for r in resp.data]
        except Exception as e:
            print(f"Supabase load_all error (falling back to file): {e}")

    return _file_load_all()


def delete(conv_id: str):
    """Delete a conversation permanently."""
    client = _get_client()
    if client:
        try:
            client.table(_TABLE).delete().eq("id", conv_id).execute()
            return
        except Exception as e:
            print(f"Supabase delete error (falling back to file): {e}")

    _file_delete(conv_id)


# ── Normalise Supabase row → standard dict ────────────────────────────

def _normalise(row: dict) -> dict:
    """Ensure messages is always a Python list (Supabase returns JSONB as list already)."""
    msgs = row.get("messages", [])
    if isinstance(msgs, str):          # shouldn't happen but guard anyway
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = []
    row["messages"] = msgs
    return row


# ── Local file fallback helpers ───────────────────────────────────────

def _file_ensure():
    os.makedirs(_LOCAL_DIR, exist_ok=True)


def _file_save(conv: dict):
    _file_ensure()
    path = os.path.join(_LOCAL_DIR, f"{conv['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


def _file_load(conv_id: str) -> Optional[dict]:
    path = os.path.join(_LOCAL_DIR, f"{conv_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _file_load_all() -> list[dict]:
    _file_ensure()
    convs = []
    for fname in os.listdir(_LOCAL_DIR):
        if fname.endswith(".json"):
            conv = _file_load(fname[:-5])
            if conv:
                convs.append(conv)
    convs.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return convs


def _file_delete(conv_id: str):
    path = os.path.join(_LOCAL_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        os.remove(path)
