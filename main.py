import os
import streamlit as st
import streamlit.components.v1 as components
import base64
import re
import io
from datetime import datetime, date, timedelta
from rag_engine import RAGEngine
import conversation_manager as cm
from fpdf import FPDF

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DOCS_DIR = os.path.join(os.getcwd(), "docs")
GDRIVE_FOLDER_ID = "1ujWlVColjvQzo6sJx7sxvdktT_u39Be1"


def _download_docs_from_drive():
    """Download PDFs from Google Drive into docs/ when the folder has no PDFs."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    existing_pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if existing_pdfs:
        return
    try:
        import gdown
        url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
        gdown.download_folder(url=url, output=DOCS_DIR, quiet=False, use_cookies=False)
    except Exception as e:
        st.warning(f"⚠️ Could not download documents from Google Drive: {e}")


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocuSense RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────
_defaults = {
    "rag": None,
    "drive_download_done": False,
    "auto_ingest_done": False,
    "current_conv": None,       # active conversation dict (None = welcome / new chat)
    "view_pdf": None,           # filename string or None
    "pdf_zoom": 100,
    "pending_delete_doc": None,
    "cached_suggested_questions": [],
    "last_ingested_count": 0,
    "pending_query": None,      # queued RAG call — set before rerun to avoid welcome flash
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# Global CSS  (light theme + chat styling)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3,h4,h5,h6 { font-family: 'Outfit', sans-serif; color: #0f172a !important; }

/* ── Main area ── */
.stApp { background-color: #f8fafc; }
.main .block-container { padding-top: 1.5rem !important; max-width: 860px !important; margin: 0 auto; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9 !important;
    border-right: 1px solid rgba(0,0,0,0.07);
}
section[data-testid="stSidebar"] > div { overflow-y: auto !important; }

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    padding: 4px 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* User bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
  div[data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 12px 16px !important;
    max-width: 82% !important;
    margin-left: auto !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
  div[data-testid="stChatMessageContent"] p {
    color: #ffffff !important;
    margin: 0 !important;
}

/* Assistant bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"])
  div[data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-left: 3px solid #6366f1 !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 14px 18px !important;
    max-width: 88% !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"])
  div[data-testid="stChatMessageContent"] p {
    color: #0f172a !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
    border-top: 1px solid rgba(0,0,0,0.08) !important;
    background: #f8fafc !important;
    padding: 10px 0 !important;
}
div[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1.5px solid rgba(99,102,241,0.35) !important;
    background: #ffffff !important;
    font-size: 1rem !important;
    padding: 12px 20px !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Buttons (general) ── */
div.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.18s ease !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#a855f7) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.35) !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.45) !important;
}
div.stButton > button[kind="secondary"] {
    background: #fff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: #f1f5f9 !important;
    border-color: #6366f1 !important;
    color: #6366f1 !important;
}

/* ═══════════════════════════════════════════
   Sidebar — conversation list
═══════════════════════════════════════════ */

/* Title column button (secondary = inactive conv) */
section[data-testid="stSidebar"] div[data-testid="column"]:first-child
  div.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    color: #475569 !important;
    font-weight: 400 !important;
    font-size: 0.78rem !important;
    border-radius: 7px !important;
    padding: 5px 8px !important;
    min-height: 28px !important;
    line-height: 1.3 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="column"]:first-child
  div.stButton > button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.09) !important;
    color: #4f46e5 !important;
    border: none !important;
    transform: none !important;
}
/* Active conversation (primary) */
section[data-testid="stSidebar"] div[data-testid="column"]:first-child
  div.stButton > button[kind="primary"] {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 7px !important;
    padding: 5px 8px !important;
    min-height: 28px !important;
    line-height: 1.3 !important;
}

/* ── Trash delete buttons (last column in sidebar rows) ── */
section[data-testid="stSidebar"] div[data-testid="column"]:last-child
  div.stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #cbd5e1 !important;
    font-size: 0.75rem !important;
    padding: 3px 5px !important;
    min-height: 26px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] div[data-testid="column"]:last-child
  div.stButton > button:hover {
    background: rgba(239,68,68,0.1) !important;
    color: #ef4444 !important;
    border: none !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── New Chat button (full-width, no column) ── */
section[data-testid="stSidebar"] > div div.stButton > button[kind="primary"] {
    text-align: center !important;
    justify-content: center !important;
}

/* ═══════════════════════════════════════════
   Welcome screen — suggested question chips
═══════════════════════════════════════════ */
.main div[data-testid="stColumn"] div.stButton > button[kind="secondary"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 7px 12px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 36px !important;
    line-height: 1.4 !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    color: #475569 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.main div[data-testid="stColumn"] div.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg,rgba(99,102,241,0.06),rgba(168,85,247,0.06)) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: #4f46e5 !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.1) !important;
    transform: translateY(-1px) !important;
}

/* ── Feedback & Action Buttons ── */
.action-btn-row {
    display: flex; gap: 8px; margin-top: 8px;
}
.action-btn-row div.stButton > button {
    background: transparent !important;
    border: none !important;
    padding: 2px 6px !important;
    font-size: 0.9rem !important;
    box-shadow: none !important;
    color: #94a3b8 !important;
    min-height: 0 !important;
    line-height: 1 !important;
}
.action-btn-row div.stButton > button:hover {
    background: rgba(0,0,0,0.05) !important;
    transform: none !important;
}

/* ── Source badges ── */
.src-badge {
    display:inline-block;background:rgba(79,70,229,0.1);
    border:1px solid rgba(79,70,229,0.25);color:#4f46e5;
    padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:600;
    margin-right:6px;margin-bottom:6px;
}
.pg-badge {
    display:inline-block;background:rgba(236,72,153,0.12);
    border:1px solid rgba(236,72,153,0.3);color:#ec4899;
    padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:600;
    margin-right:6px;margin-bottom:6px;
}

/* ── PDF viewer ── */
.viewer-header {
    display:flex;justify-content:space-between;align-items:center;
    background:#fff;border:1px solid rgba(0,0,0,0.1);border-radius:14px;
    padding:12px 18px;margin-bottom:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
}
.viewer-chip {
    background:rgba(34,197,94,0.14);border:1px solid rgba(34,197,94,0.35);
    color:#22c55e;border-radius:999px;font-size:0.72rem;font-weight:700;padding:3px 12px;
}
.viewer-doc-name {font-size:1rem;font-weight:700;color:#0f172a;}
.viewer-zoom-indicator {
    color:#64748b;font-size:0.82rem;font-weight:700;
    background:#f8fafc;border:1px solid rgba(0,0,0,0.1);border-radius:8px;padding:4px 12px;
}

/* ── Date group labels ── */
.grp-label {
    font-size:0.72rem;color:#94a3b8;font-weight:700;
    letter-spacing:0.06em;text-transform:uppercase;
    padding:10px 10px 2px 10px;
}

/* ── Welcome screen ── */
.welcome-wrap {text-align:center;padding:3rem 1rem 1.5rem 1rem;}
.welcome-title {
    background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:2.8rem;font-weight:800;font-family:'Outfit',sans-serif;
    margin-bottom:0.4rem;
}
.welcome-sub {color:#64748b;font-size:1.1rem;margin:0;}

/* ── Scrollbar ── */
::-webkit-scrollbar {width:6px;height:6px;}
::-webkit-scrollbar-track {background:#f1f5f9;}
::-webkit-scrollbar-thumb {background:#cbd5e1;border-radius:4px;}
::-webkit-scrollbar-thumb:hover {background:#94a3b8;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Startup: Drive download → RAGEngine → Auto-ingest
# ──────────────────────────────────────────────
if not st.session_state.drive_download_done:
    existing_pdfs = (
        [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
        if os.path.isdir(DOCS_DIR) else []
    )
    if not existing_pdfs:
        with st.spinner("☁️ Downloading documents from Google Drive..."):
            _download_docs_from_drive()
    st.session_state.drive_download_done = True

if st.session_state.rag is None:
    with st.spinner("⚙️ Initialising RAG engine..."):
        st.session_state.rag = RAGEngine()

rag: RAGEngine = st.session_state.rag

if not st.session_state.auto_ingest_done:
    pdf_count = (
        len([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])
        if os.path.isdir(DOCS_DIR) else 0
    )
    if pdf_count > 0 and rag.collection.count() == 0:
        with st.spinner("📚 Indexing documents into ChromaDB..."):
            rag.ingest_documents(DOCS_DIR)
    st.session_state.auto_ingest_done = True

ingested_files = rag.get_ingested_files()

# Refresh suggested questions when doc count changes
if len(ingested_files) != st.session_state.last_ingested_count:
    st.session_state.cached_suggested_questions = rag.get_suggested_questions(DOCS_DIR)
    st.session_state.last_ingested_count = len(ingested_files)
elif not st.session_state.cached_suggested_questions and ingested_files:
    st.session_state.cached_suggested_questions = rag.get_suggested_questions(DOCS_DIR)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _group_conversations(convs: list[dict]) -> dict[str, list]:
    today = date.today()
    yesterday = today - timedelta(days=1)
    week_ago  = today - timedelta(days=7)
    groups: dict[str, list] = {
        "Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": []
    }
    for conv in convs:
        try:
            d = datetime.fromisoformat(conv["updated_at"]).date()
        except Exception:
            d = today
        if d == today:
            groups["Today"].append(conv)
        elif d == yesterday:
            groups["Yesterday"].append(conv)
        elif d > week_ago:
            groups["Previous 7 Days"].append(conv)
        else:
            groups["Older"].append(conv)
    return groups


def _start_new_chat():
    st.session_state.current_conv = None
    st.session_state.view_pdf = None

def _handle_suggested_question(q: str):
    new_conv = cm.new_conversation()
    new_conv["title"] = q[:50]
    new_conv["messages"].append({
        "role": "user", "content": q,
        "timestamp": datetime.now().isoformat()
    })
    st.session_state.current_conv = new_conv
    st.session_state.pending_query = q


def _export_txt(msgs: list) -> str:
    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "DocuSense"
        lines.append(f"{role}:\n{m['content']}\n")
        if role == "DocuSense" and m.get("citations"):
            lines.append("Sources:")
            for c in m["citations"]:
                lines.append(f"  - {c.get('source', '')} (Page {c.get('page', '')})")
            lines.append("")
        lines.append("-" * 40 + "\n")
    return "\n".join(lines)


def _export_pdf(msgs: list, title: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "C:\\Windows\\Fonts\\arial.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)
    
    pdf.set_font("DejaVu", style="B", size=16)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.set_font("DejaVu", size=12)
    pdf.ln(10)
    
    for m in msgs:
        role = "User" if m["role"] == "user" else "DocuSense"
        pdf.set_font("DejaVu", style="B", size=12)
        pdf.multi_cell(0, 10, txt=f"{role}:")
        pdf.set_font("DejaVu", size=11)
        # remove emojis which FPDF might struggle with
        content = m['content']
        pdf.multi_cell(0, 8, txt=content)
        if role == "DocuSense" and m.get("citations"):
            pdf.ln(2)
            pdf.set_font("DejaVu", style="I", size=10)
            pdf.multi_cell(0, 6, txt="Sources:")
            for c in m["citations"]:
                pdf.multi_cell(0, 6, txt=f"  - {c.get('source', '')} (Page {c.get('page', '')})")
        pdf.ln(5)
        pdf.set_font("DejaVu", size=10)
        pdf.cell(0, 0, txt="-" * 60, ln=True)
        pdf.ln(5)
    
    return pdf.output(dest="S").encode("latin1", "replace")  # output to string and encode

def _open_conv(conv_id: str):
    conv = cm.load(conv_id)
    if conv:
        st.session_state.current_conv = conv
        st.session_state.view_pdf = None


def _open_pdf(filename: str):
    st.session_state.view_pdf = filename


def _render_citations(citations: list, sources: list):
    """Render source citations inside an expander."""
    if not citations:
        return
    with st.expander(f"📄 {len(citations)} source(s)", expanded=False):
        for cit in citations:
            src   = cit.get("source", "")
            page  = cit.get("page", "")
            quote = cit.get("exact_quote", "")
            st.markdown(
                f'<span class="src-badge">📄 {src}</span>'
                f'<span class="pg-badge">Page {page}</span>',
                unsafe_allow_html=True
            )
            if quote:
                # Highlight the exact quote in the source chunk
                best_text = quote
                for s in sources:
                    if s.get("source", "").lower() == src.lower():
                        chunk = s.get("text", "")
                        if quote.lower() in chunk.lower():
                            try:
                                highlighted = re.compile(re.escape(quote), re.IGNORECASE).sub(
                                    lambda m: f"**{m.group(0)}**", chunk
                                )
                                best_text = highlighted
                            except Exception:
                                best_text = chunk
                        break
                st.markdown(f"> {best_text}")
                st.markdown("")


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    # Branding
    col_ic, col_tx = st.columns([1, 3])
    with col_ic:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=42)
    with col_tx:
        st.markdown(
            "<h2 style='margin:0;padding-top:6px;font-size:1.1rem;color:#0f172a;'>"
            "DocuSense RAG</h2>"
            "<p style='margin:0;font-size:0.78rem;color:#64748b;'>ChromaDB · Groq</p>",
            unsafe_allow_html=True
        )

    st.write("")

    # New Chat
    if st.button("✏️  New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
        _start_new_chat()
        st.rerun()

    st.write("")

    # ── Export Chat ──
    if st.session_state.current_conv and st.session_state.current_conv.get("messages"):
        with st.expander("📥 Export Current Chat"):
            msgs = st.session_state.current_conv["messages"]
            title = st.session_state.current_conv.get("title", "Conversation")
            
            txt_data = _export_txt(msgs)
            st.download_button("Export as TXT", data=txt_data, file_name=f"{title}.txt", mime="text/plain", use_container_width=True)
            
            try:
                pdf_data = _export_pdf(msgs, title)
                st.download_button("Export as PDF", data=pdf_data, file_name=f"{title}.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF export error: {e}")

    # ── Conversation history ──
    search_query = st.text_input("🔍 Search chats...", key="chat_search", label_visibility="collapsed", placeholder="🔍 Search chats...")
    
    all_convs = cm.load_all()
    if search_query:
        all_convs = [c for c in all_convs if search_query.lower() in c.get("title", "").lower()]

    active_id = (st.session_state.current_conv or {}).get("id")

    if all_convs:
        groups = _group_conversations(all_convs)
        for grp_name, grp_convs in groups.items():
            if not grp_convs:
                continue
            st.markdown(f"<div class='grp-label'>{grp_name}</div>", unsafe_allow_html=True)
            for conv in grp_convs:
                    is_active = conv["id"] == active_id
                    label = conv.get("title", "New Chat")
                    label_display = label if len(label) <= 34 else label[:32] + "…"
                    btn_type = "primary" if is_active else "secondary"
                    icon = "● " if is_active else "  "

                    c_title, c_del = st.columns([6, 1], gap="small")
                    with c_title:
                        if st.button(
                            f"{icon}{label_display}",
                            key=f"conv_btn_{conv['id']}",
                            use_container_width=True,
                            type=btn_type
                        ):
                            _open_conv(conv["id"])
                            st.rerun()
                    with c_del:
                        if st.button(
                            "🗑",
                            key=f"del_conv_{conv['id']}",
                            help="Delete this conversation",
                            use_container_width=True,
                            type="secondary"
                        ):
                            cm.delete(conv["id"])
                            # If we just deleted the active conversation, go to welcome screen
                            if st.session_state.current_conv and \
                               st.session_state.current_conv.get("id") == conv["id"]:
                                st.session_state.current_conv = None
                            st.rerun()
    else:
        st.caption("No conversations yet. Start typing below!")

    st.divider()

    # ── Documents ──
    doc_label = f"📄 Documents ({len(ingested_files)})"
    with st.expander(doc_label, expanded=False):
        if ingested_files:
            for idx, fname in enumerate(ingested_files):
                c1, c2 = st.columns([5, 1])
                with c1:
                    if st.button(
                        f"📄 {fname}", key=f"view_pdf_{idx}",
                        use_container_width=True, type="secondary"
                    ):
                        _open_pdf(fname)
                        st.rerun()
                with c2:
                    if st.button("🗑", key=f"del_doc_{idx}", help=f"Delete {fname}"):
                        st.session_state.pending_delete_doc = fname
                        st.rerun()
        else:
            st.caption("No documents indexed yet.")
            if st.button("🚀 Index PDFs", key="sidebar_ingest_btn",
                         use_container_width=True, type="primary"):
                with st.spinner("Indexing..."):
                    rag.ingest_documents(DOCS_DIR)
                st.session_state.auto_ingest_done = True
                st.rerun()

    st.markdown(
        "<div style='color:#94a3b8;font-size:0.72rem;text-align:center;padding-top:6px;'>"
        "Powered by ChromaDB &amp; Groq API</div>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# Delete-doc confirmation dialog
# ──────────────────────────────────────────────
if st.session_state.pending_delete_doc:
    @st.dialog("Delete Document?")
    def _confirm_delete():
        target = st.session_state.pending_delete_doc
        st.warning(
            f"This will permanently remove **{target}** and rebuild the index."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Delete", use_container_width=True,
                         type="primary", key="confirm_del_btn"):
                del_path = os.path.join(DOCS_DIR, target)
                try:
                    if os.path.exists(del_path):
                        os.remove(del_path)
                    remaining = [
                        f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")
                    ]
                    if remaining:
                        rag.ingest_documents(DOCS_DIR)
                    else:
                        try:
                            rag.chroma_client.delete_collection("pdf_documents")
                        except Exception:
                            pass
                        rag.collection = rag.chroma_client.get_or_create_collection(
                            name="pdf_documents",
                            embedding_function=rag.embedding_function
                        )
                except Exception as e:
                    st.error(str(e))
                st.session_state.pending_delete_doc = None
                st.session_state.auto_ingest_done = True
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True, key="cancel_del_btn"):
                st.session_state.pending_delete_doc = None
                st.rerun()

    _confirm_delete()

# ══════════════════════════════════════════════
#  MAIN CONTENT AREA
# ══════════════════════════════════════════════

# ── PDF Viewer ──────────────────────────────
if st.session_state.view_pdf:
    selected_doc = st.session_state.view_pdf
    pdf_path = os.path.join(DOCS_DIR, selected_doc)

    if os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            st.markdown(
                f"""
                <div class="viewer-header">
                  <div style="display:flex;align-items:center;gap:12px;">
                    <span class="viewer-chip">PDF</span>
                    <span class="viewer-doc-name">{selected_doc}</span>
                  </div>
                  <span class="viewer-zoom-indicator">{st.session_state.pdf_zoom}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            t1, t2, t3, t4, t5 = st.columns([2.2, 1, 1, 1, 2.2], gap="small")
            with t1:
                if st.button("⬅ Back to Chat", key="back_btn", use_container_width=True):
                    st.session_state.view_pdf = None
                    st.rerun()
            with t2:
                if st.button("－ Zoom", key="zoom_out", use_container_width=True, type="secondary"):
                    st.session_state.pdf_zoom = max(50, st.session_state.pdf_zoom - 25)
                    st.rerun()
            with t3:
                if st.button("＋ Zoom", key="zoom_in", use_container_width=True, type="secondary"):
                    st.session_state.pdf_zoom = min(300, st.session_state.pdf_zoom + 25)
                    st.rerun()
            with t4:
                if st.button("Reset", key="zoom_reset", use_container_width=True, type="secondary"):
                    st.session_state.pdf_zoom = 100
                    st.rerun()
            with t5:
                st.download_button(
                    "⬇ Download", data=pdf_bytes, file_name=selected_doc,
                    mime="application/pdf", use_container_width=True, key="dl_pdf"
                )

            zoom  = st.session_state.pdf_zoom
            scale = round(zoom / 100.0, 2)
            pdf_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#e5e7eb;}}
  #lbl{{text-align:center;padding:32px;color:#6b7280;font:13px sans-serif;}}
  #box{{display:flex;flex-direction:column;align-items:center;gap:10px;padding:12px;}}
  canvas{{background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.18);border-radius:4px;display:block;}}
</style></head><body>
<div id="lbl">⏳ Rendering PDF...</div><div id="box"></div>
<script>
pdfjsLib.GlobalWorkerOptions.workerSrc=
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
var b64="{base64_pdf}",bin=atob(b64),buf=new Uint8Array(bin.length);
for(var i=0;i<bin.length;i++)buf[i]=bin.charCodeAt(i);
pdfjsLib.getDocument({{data:buf}}).promise.then(function(pdf){{
  document.getElementById('lbl').style.display='none';
  var box=document.getElementById('box'),sc={scale},chain=Promise.resolve();
  for(var p=1;p<=pdf.numPages;p++){{
    (function(n){{chain=chain.then(function(){{return pdf.getPage(n).then(function(pg){{
      var vp=pg.getViewport({{scale:sc}}),c=document.createElement('canvas');
      c.width=vp.width;c.height=vp.height;box.appendChild(c);
      return pg.render({{canvasContext:c.getContext('2d'),viewport:vp}}).promise;
    }})}})}})(p);
  }}
}}).catch(function(e){{document.getElementById('lbl').textContent='❌ '+e.message;}});
</script></body></html>"""
            components.html(pdf_html, height=800, scrolling=True)

        except Exception as e:
            st.error(f"Error loading PDF: {e}")
    else:
        st.error("Document file not found.")

# ── Chat Mode ───────────────────────────────
else:
    conv = st.session_state.current_conv
    msgs = conv["messages"] if conv else []

    # ── Declare chat_input FIRST so its value suppresses the welcome screen instantly ──
    if ingested_files:
        user_input = st.chat_input(
            "Message DocuSense — ask a question or just say hi!",
            key="chat_input_main"
        )
    else:
        user_input = None

    # ── Welcome screen: only when no messages AND no active submission in progress ──
    if not msgs and not user_input and not st.session_state.pending_query:
        st.markdown("""
        <div class="welcome-wrap">
            <h1 class="welcome-title">DocuSense RAG</h1>
            <p class="welcome-sub">Query your documents with AI-powered precision.</p>
        </div>
        """, unsafe_allow_html=True)

        if not ingested_files:
            st.info(
                "No documents indexed yet. Open **📄 Documents** in the sidebar "
                "to index your PDFs."
            )
        else:
            sq = st.session_state.cached_suggested_questions
            if sq:
                st.markdown(
                    "<p style='text-align:center;color:#64748b;font-size:0.85rem;"
                    "margin-bottom:12px;'>💡 <b>Try asking:</b></p>",
                    unsafe_allow_html=True
                )
                cols = st.columns(2)
                for i, q in enumerate(sq):
                    with cols[i % 2]:
                        st.button(
                            q, 
                            key=f"sq_{i}", 
                            use_container_width=True, 
                            type="secondary",
                            on_click=_handle_suggested_question,
                            args=(q,)
                        )

    # ── Render existing conversation messages ──
    elif msgs:
        for idx, msg in enumerate(msgs):
            role   = msg["role"]
            avatar = "👤" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                if role == "assistant":
                    _render_citations(
                        msg.get("citations", []),
                        msg.get("sources", [])
                    )
                    
                    st.markdown("<div class='action-btn-row'>", unsafe_allow_html=True)
                    c1, c2, c3, _ = st.columns([1, 1, 1, 10])
                    with c1:
                        if st.button("📋", key=f"copy_{idx}", help="Copy to clipboard"):
                            st.toast("Answer ready to copy! (Highlight text and use Ctrl+C)", icon="📋")
                    with c2:
                        up_color = "🟢" if msg.get("feedback") == "up" else "👍"
                        if st.button(up_color, key=f"up_{idx}", help="Good response"):
                            msg["feedback"] = "up"
                            cm.save(st.session_state.current_conv)
                            st.rerun()
                    with c3:
                        down_color = "🔴" if msg.get("feedback") == "down" else "👎"
                        if st.button(down_color, key=f"down_{idx}", help="Bad response"):
                            msg["feedback"] = "down"
                            cm.save(st.session_state.current_conv)
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    # ── Process pending RAG query (runs after messages are shown) ──
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

        # Build chat history = everything BEFORE the current user message
        # (the current user msg is the last item in msgs)
        conv_msgs = (st.session_state.current_conv or {}).get("messages", [])
        chat_history = conv_msgs[:-1] if len(conv_msgs) > 1 else []

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                resp = rag.smart_query(query, chat_history=chat_history)
            intent    = resp.get("intent", "document_query")
            answer    = resp.get("answer", "")
            citations = resp.get("citations", [])
            sources   = resp.get("sources", [])
            st.markdown(answer)
            if intent == "document_query" and "answer not in documents" not in answer.lower():
                _render_citations(citations, sources)
        if st.session_state.current_conv:
            st.session_state.current_conv["messages"].append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
            
            # Smart Title Generation on first exchange
            if len(st.session_state.current_conv["messages"]) == 2:
                new_title = rag.generate_title(query, answer)
                st.session_state.current_conv["title"] = new_title
                
            cm.save(st.session_state.current_conv)
        st.rerun()

    # ── Handle new user message from chat_input ──
    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        if st.session_state.current_conv is None:
            new_conv = cm.new_conversation()
            new_conv["title"] = user_input[:50]
            st.session_state.current_conv = new_conv

        st.session_state.current_conv["messages"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.pending_query = user_input
        cm.save(st.session_state.current_conv)   # save with just the user message
        st.rerun()

    if not ingested_files:
        st.info(
            "💡 Open **📄 Documents** in the sidebar and click **🚀 Index PDFs** "
            "to start chatting."
        )
