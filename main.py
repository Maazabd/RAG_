import os
import streamlit as st
import base64
import re
from rag_engine import RAGEngine

# Set Page Config
st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme selection initialization
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Shared UI constants/state defaults
DOC_DEFAULT_OPTION = "-- Select a Document to View --"
if "return_to_qa" not in st.session_state:
    st.session_state.return_to_qa = False

# Handle return action before creating sidebar widgets so selectbox state can be safely reset
if st.session_state.return_to_qa:
    st.session_state.selected_doc_name = DOC_DEFAULT_OPTION
    st.session_state.return_to_qa = False

# Define theme variables
if st.session_state.theme == "dark":
    BG_COLOR = "#080c14"
    TEXT_COLOR = "#f8fafc"
    SUB_TEXT_COLOR = "#94a3b8"
    CARD_BG = "rgba(30, 41, 59, 0.45)"
    CARD_BORDER = "rgba(255, 255, 255, 0.12)"
    CARD_SHADOW = "rgba(0, 0, 0, 0.45)"
    INPUT_BG = "#151e2e"
    INPUT_COLOR = "#ffffff"
    INPUT_BORDER = "rgba(255, 255, 255, 0.2)"
    SIDEBAR_BG = "#0b0e17"
    SIDEBAR_BORDER = "rgba(255, 255, 255, 0.08)"
    HEADER_GRADIENT = "linear-gradient(135deg, rgba(20, 25, 40, 0.85) 0%, rgba(8, 12, 20, 0.95) 100%)"
    BADGE_BG = "rgba(99, 102, 241, 0.25)"
    BADGE_BORDER = "rgba(99, 102, 241, 0.45)"
    BADGE_COLOR = "#cbd5e1"
    # Button colors for high readability in dark mode without hovering
    BTN_BG = "#3b82f6" 
    BTN_TEXT = "#ffffff"
    BTN_BORDER = "transparent"
else:
    BG_COLOR = "#f8fafc"
    TEXT_COLOR = "#0f172a"
    SUB_TEXT_COLOR = "#475569"
    CARD_BG = "rgba(255, 255, 255, 0.95)"
    CARD_BORDER = "rgba(0, 0, 0, 0.1)"
    CARD_SHADOW = "rgba(0, 0, 0, 0.06)"
    INPUT_BG = "#ffffff"
    INPUT_COLOR = "#0f172a"
    INPUT_BORDER = "rgba(0, 0, 0, 0.2)"
    SIDEBAR_BG = "#f1f5f9"
    SIDEBAR_BORDER = "rgba(0, 0, 0, 0.08)"
    HEADER_GRADIENT = "linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.98) 100%)"
    BADGE_BG = "rgba(79, 70, 229, 0.1)"
    BADGE_BORDER = "rgba(79, 70, 229, 0.25)"
    BADGE_COLOR = "#4f46e5"
    BTN_BG = "#f1f5f9"
    BTN_TEXT = "#0f172a"
    BTN_BORDER = "#cbd5e1"

# Apply theme CSS styles dynamically
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;700&display=swap');

    /* Font applications */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Outfit', sans-serif;
        color: {TEXT_COLOR} !important;
    }}

    /* Main Background & Text Color */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}

    /* Title Styling with Custom Gradient */
    .title-container {{
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        background: {HEADER_GRADIENT};
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid {CARD_BORDER};
        box-shadow: 0 8px 32px 0 {CARD_SHADOW};
    }}
    .main-title {{
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }}
    .sub-title {{
        color: {SUB_TEXT_COLOR};
        font-size: 1.15rem;
        font-weight: 400;
    }}

    /* High contrast text overrides */
    div[data-testid="stMarkdownContainer"] p {{
        color: {TEXT_COLOR} !important;
        font-size: 1.02rem;
        line-height: 1.6;
    }}
    div[data-testid="stMarkdownContainer"] li {{
        color: {TEXT_COLOR} !important;
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: {CARD_BG};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 {CARD_SHADOW};
        color: {TEXT_COLOR} !important;
    }}
    .answer-card {{
        background: {CARD_BG};
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.25);
        color: {TEXT_COLOR} !important;
    }}
    .not-found-card {{
        background: {CARD_BG};
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(239, 68, 68, 0.15);
        color: {TEXT_COLOR} !important;
    }}

    /* Source Badges */
    .source-badge {{
        display: inline-block;
        background-color: {BADGE_BG};
        border: 1px solid {BADGE_BORDER};
        color: {BADGE_COLOR};
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }}
    .page-badge {{
        display: inline-block;
        background-color: rgba(236, 72, 153, 0.15);
        border: 1px solid rgba(236, 72, 153, 0.35);
        color: #ec4899;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }}

    /* Input Field Styling */
    .stTextInput>div>div>input {{
        background-color: {INPUT_BG} !important;
        color: {INPUT_COLOR} !important;
        border: 1px solid {INPUT_BORDER} !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        padding: 12px 18px !important;
        transition: all 0.3s ease;
    }}
    .stTextInput>div>div>input:focus {{
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }}

    /* High-contrast explicit button styling */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
        background-color: {BTN_BG} !important;
        color: {BTN_TEXT} !important;
        border: 1px solid {BTN_BORDER} !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }}
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid {SIDEBAR_BORDER};
    }}
    section[data-testid="stSidebar"] > div {{
        overflow-y: hidden !important;
        height: 100vh !important;
    }}
    
    /* Sidebar text color visibility */
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {TEXT_COLOR} !important;
    }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BG_COLOR};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {INPUT_BG};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #334155;
    }}

    /* Highlight marking style */
    mark {{
        border-radius: 4px;
        padding: 1px 4px;
    }}

    /* Modern PDF viewer layout */
    .viewer-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        background: {CARD_BG};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 6px 22px -14px {CARD_SHADOW};
    }}
    .viewer-header-left {{
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
    }}
    .viewer-chip {{
        flex: 0 0 auto;
        background: rgba(34, 197, 94, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #22c55e;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 12px;
        letter-spacing: 0.04em;
    }}
    .viewer-doc-name {{
        color: {TEXT_COLOR};
        font-size: 1.02rem;
        font-weight: 700;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
    }}
    .viewer-zoom-indicator {{
        flex: 0 0 auto;
        color: {SUB_TEXT_COLOR};
        font-size: 0.82rem;
        font-weight: 700;
        background: {INPUT_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 4px 12px;
        font-variant-numeric: tabular-nums;
    }}
    .viewer-frame-wrap {{
        background: {CARD_BG};
        padding: 14px;
        border-radius: 16px;
        margin-top: 8px;
        border: 1px solid {CARD_BORDER};
        box-shadow: 0 12px 34px -20px {CARD_SHADOW};
    }}
    .viewer-frame-inner {{
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 2px 10px -4px rgba(15, 23, 42, 0.35);
    }}
    .viewer-frame {{
        display: block;
        width: 100%;
        height: 82vh;
        min-height: 560px;
        border: none;
        background: #ffffff;
    }}

    /* Sidebar document card and dialog accents */
    .doc-card-hint {{
        margin: 0 0 10px 0;
        color: {SUB_TEXT_COLOR};
        font-size: 0.78rem;
        opacity: 0.9;
    }}
    .doc-row-card {{
        margin-bottom: 8px;
    }}

    /* Keep icon buttons compact so they stay visually inside row cards */
    div.stButton > button[kind="tertiary"] {{
        padding: 6px 10px !important;
        min-height: 34px !important;
    }}
    .dialog-shell {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
    }}
    .dialog-chip {{
        display: inline-block;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 999px;
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #ef4444;
        margin-bottom: 8px;
    }}
    .dialog-title {{
        margin: 0;
        font-size: 1.03rem;
        font-weight: 700;
        color: {TEXT_COLOR};
    }}
    .dialog-sub {{
        margin: 6px 0 0 0;
        color: {SUB_TEXT_COLOR};
        font-size: 0.86rem;
        line-height: 1.45;
    }}

    @media (max-width: 768px) {{
        .viewer-shell {{
            padding: 12px;
            border-radius: 14px;
        }}
        .viewer-name {{
            font-size: 1.02rem;
            line-height: 1.35;
        }}
        .viewer-frame {{
            height: 72vh;
            min-height: 420px;
        }}
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session State for RAGEngine
if "rag" not in st.session_state:
    with st.spinner("Initializing Chroma DB & Embedding Models..."):
        st.session_state.rag = RAGEngine()

rag = st.session_state.rag

if "selected_doc_name" not in st.session_state:
    st.session_state.selected_doc_name = DOC_DEFAULT_OPTION

if "pending_delete_doc" not in st.session_state:
    st.session_state.pending_delete_doc = None

if "sidebar_feedback" not in st.session_state:
    st.session_state.sidebar_feedback = None

if "pdf_zoom" not in st.session_state:
    st.session_state.pdf_zoom = 100

if "active_pdf_doc" not in st.session_state:
    st.session_state.active_pdf_doc = None

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.markdown("<h2 style='margin-top: 0.5rem; margin-bottom: 0px;'>DocuSense RAG</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 0.85rem; margin-bottom: 1rem;'>ChromaDB + Groq Assistant</p>", unsafe_allow_html=True)
    
    # ☀️/🌙 Mode switcher shifted to the top of the sidebar
    theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    theme_label = "Light Mode" if st.session_state.theme == "dark" else "Dark Mode"
    if st.button(f"{theme_icon} Switch to {theme_label}", key="theme_toggle_btn", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
        
    st.write("---")

    # Document list with per-file actions
    st.markdown("### 📁 Document Index")
    ingested_files = rag.get_ingested_files()

    # Keep selected document consistent with current index state
    if st.session_state.selected_doc_name not in ingested_files:
        st.session_state.selected_doc_name = DOC_DEFAULT_OPTION

    selected_doc = st.session_state.selected_doc_name

    feedback = st.session_state.sidebar_feedback
    if feedback:
        if feedback["kind"] == "success":
            st.success(feedback["msg"])
        elif feedback["kind"] == "error":
            st.error(feedback["msg"])
        else:
            st.warning(feedback["msg"])
        st.session_state.sidebar_feedback = None

    if ingested_files:
        st.markdown("<p class='doc-card-hint'>Click a file name to open it. Use the trash icon to delete.</p>", unsafe_allow_html=True)
        for idx, file_name in enumerate(ingested_files):
            with st.container(border=True):
                c_name, c_del = st.columns([10, 1], gap="small")
                with c_name:
                    is_active = file_name == selected_doc
                    file_label = f"● {file_name}" if is_active else file_name
                    if st.button(file_label, key=f"open_doc_{idx}", use_container_width=True, type="tertiary"):
                        st.session_state.selected_doc_name = file_name
                        st.rerun()
                with c_del:
                    if st.button("🗑", key=f"delete_icon_{idx}", help=f"Delete {file_name}", type="tertiary"):
                        st.session_state.pending_delete_doc = file_name
                        st.rerun()
    else:
        st.warning("No documents indexed yet.")

    st.write("---")
    st.markdown(
        "<div style='color: #64748b; font-size: 0.8rem; text-align: center;'>"
        "Powered by ChromaDB & Groq API"
        "</div>",
        unsafe_allow_html=True
    )

# Confirmation popup for deleting a document
if st.session_state.pending_delete_doc:
    @st.dialog("Delete Document?")
    def confirm_delete_dialog():
        target = st.session_state.pending_delete_doc
        st.markdown(
            f"""
            <div class="dialog-shell">
                <span class="dialog-chip">DELETE CONFIRMATION</span>
                <p class="dialog-title">Remove <strong>{target}</strong> from workspace?</p>
                <p class="dialog-sub">This action deletes the file and refreshes the document index.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Delete Permanently", use_container_width=True, type="primary", key="confirm_delete_btn"):
                delete_path = os.path.join(os.getcwd(), target)
                if not os.path.exists(delete_path):
                    st.session_state.sidebar_feedback = {"kind": "warning", "msg": "Selected file was not found on disk."}
                else:
                    try:
                        os.remove(delete_path)
                    except Exception as e:
                        st.session_state.sidebar_feedback = {"kind": "error", "msg": f"Could not delete '{target}': {e}"}
                    else:
                        if st.session_state.selected_doc_name == target:
                            st.session_state.return_to_qa = True

                        with st.spinner("Updating index after file deletion..."):
                            remaining_pdfs = [
                                f for f in os.listdir(os.getcwd())
                                if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(os.getcwd(), f))
                            ]

                            if remaining_pdfs:
                                res = rag.ingest_documents(os.getcwd())
                                if res["status"] == "success":
                                    st.session_state.sidebar_feedback = {"kind": "success", "msg": f"Deleted '{target}' and refreshed index."}
                                else:
                                    st.session_state.sidebar_feedback = {"kind": "error", "msg": res["message"]}
                            else:
                                try:
                                    rag.chroma_client.delete_collection("pdf_documents")
                                except Exception:
                                    pass

                                rag.collection = rag.chroma_client.get_or_create_collection(
                                    name="pdf_documents",
                                    embedding_function=rag.embedding_function
                                )

                                suggested_path = os.path.join(os.getcwd(), "suggested_questions.json")
                                if os.path.exists(suggested_path):
                                    try:
                                        os.remove(suggested_path)
                                    except Exception:
                                        pass

                                st.session_state.sidebar_feedback = {"kind": "success", "msg": f"Deleted '{target}'. No PDFs remain in workspace."}

                st.session_state.pending_delete_doc = None
                st.rerun()
        with c2:
            if st.button("Keep File", use_container_width=True, key="cancel_delete_btn"):
                st.session_state.pending_delete_doc = None
                st.rerun()

    confirm_delete_dialog()

# Focus mode CSS when viewing a document
if selected_doc != DOC_DEFAULT_OPTION:
    st.markdown(
        """
        <style>
            html, body, [data-testid="stAppViewContainer"], .stApp, .main {
                height: 100vh !important;
                overflow: hidden !important;
            }
            section[data-testid="stSidebar"] { display: none !important; }
            div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
            .main .block-container {
                max-width: 100% !important;
                height: 100vh !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                overflow: hidden !important;
            }
            .viewer-frame-wrap {
                margin-top: 0 !important;
                height: calc(100vh - 128px) !important;
            }
            .viewer-frame-inner {
                height: 100% !important;
            }
            .viewer-frame {
                height: 100% !important;
                min-height: 100% !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Check if document view mode is selected from dropdown list
if selected_doc != DOC_DEFAULT_OPTION:
    # Dedicated screen for PDF Viewer
    pdf_path = os.path.join(os.getcwd(), selected_doc)
    if os.path.exists(pdf_path):
        try:
            if st.session_state.active_pdf_doc != selected_doc:
                st.session_state.active_pdf_doc = selected_doc
                st.session_state.pdf_zoom = 100

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

            st.markdown(
                f"""
                <div class="viewer-header">
                    <div class="viewer-header-left">
                        <span class="viewer-chip">PDF</span>
                        <span class="viewer-doc-name">{selected_doc}</span>
                    </div>
                    <span class="viewer-zoom-indicator">{st.session_state.pdf_zoom}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            t1, t2, t3, t4, t5 = st.columns([2.1, 1, 1, 1, 2.1], gap="small")
            with t1:
                if st.button("⬅ Back", key="back_to_qa_btn", use_container_width=True):
                    st.session_state.return_to_qa = True
                    st.rerun()
            with t2:
                if st.button("－ Zoom", key="zoom_out_btn", use_container_width=True, type="tertiary"):
                    st.session_state.pdf_zoom = max(50, st.session_state.pdf_zoom - 25)
                    st.rerun()
            with t3:
                if st.button("＋ Zoom", key="zoom_in_btn", use_container_width=True, type="tertiary"):
                    st.session_state.pdf_zoom = min(300, st.session_state.pdf_zoom + 25)
                    st.rerun()
            with t4:
                if st.button("Reset", key="zoom_reset_btn", use_container_width=True, type="tertiary"):
                    st.session_state.pdf_zoom = 100
                    st.rerun()
            with t5:
                st.download_button(
                    "⬇ Download",
                    data=pdf_bytes,
                    file_name=selected_doc,
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_open_pdf"
                )

            # Use the PDF viewer's native zoom parameter for reliable, crisp scaling
            pdf_src = (
                f"data:application/pdf;base64,{base64_pdf}"
                f"#toolbar=0&navpanes=0&scrollbar=0&zoom={st.session_state.pdf_zoom}"
            )

            pdf_display = (
                f'<div class="viewer-frame-wrap"><div class="viewer-frame-inner">'
                f'<iframe class="viewer-frame" src="{pdf_src}"></iframe>'
                f'</div></div>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
    else:
        st.error("Document file not found.")

# If in Q&A Mode
else:
    # Main Application Dashboard
    st.markdown(
        """
        <div class="title-container">
            <h1 class="main-title">DocuSense RAG Engine</h1>
            <p class="sub-title">Query your workspace PDF documents with precise, context-bounded AI answers.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # If no documents are indexed, prompt ingestion
    if not ingested_files:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <h3 style="color: #f59e0b; margin-bottom: 15px;">⚠️ Ingestion Required</h3>
                <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 25px;">
                    ChromaDB is empty. We found PDF documents in your workspace directory. 
                    Please click below to parse, chunk, embed, and index them.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Ingest & Index PDFs Now", use_container_width=True, type="primary"):
                with st.spinner("Processing documents (generating embeddings)..."):
                    res = rag.ingest_documents(os.getcwd())
                    if res["status"] == "success":
                        st.success(res["message"])
                        st.rerun()
                    else:
                        st.error(res["message"])
    else:
        # Q&A Section
        st.markdown("### 💬 Ask a Question")

        def _trigger_query_run():
            st.session_state.run_query_now = True

        def _clear_query_text():
            st.session_state.query_input = ""
            st.session_state.run_query_now = False
        
        # Handle sample query state
        if "selected_sample_query" not in st.session_state:
            st.session_state.selected_sample_query = None

        if "query_input" not in st.session_state:
            st.session_state.query_input = ""

        if "run_query_now" not in st.session_state:
            st.session_state.run_query_now = False

        auto_trigger = False
        if st.session_state.selected_sample_query:
            st.session_state.query_input = st.session_state.selected_sample_query
            auto_trigger = True
            st.session_state.selected_sample_query = None # clear immediately

        query_col, clear_col = st.columns([14, 1], gap="small")
        with query_col:
            st.text_input(
                "Enter your question:",
                key="query_input",
                placeholder="Type your question and press Enter...",
                label_visibility="collapsed",
                on_change=_trigger_query_run
            )
        with clear_col:
            st.button(
                "✕",
                key="clear_query_btn",
                help="Clear query",
                use_container_width=True,
                type="tertiary",
                on_click=_clear_query_text
            )

        # Determine if we run the query
        query_text = st.session_state.query_input.strip()
        should_run = False
        if auto_trigger and query_text:
            should_run = True
        elif st.session_state.run_query_now:
            should_run = bool(query_text)
            st.session_state.run_query_now = False

        # 1. First display search results (ABOVE the suggested questions)
        if should_run:
            if not query_text.strip():
                st.warning("Please enter a valid query.")
            else:
                with st.spinner("Retrieving document chunks and reasoning..."):
                    response = rag.query(query_text)
                    
                answer = response["answer"]
                citations = response.get("citations", [])
                sources = response.get("sources", [])
                
                # Style answer card appropriately
                if "answer not in documents" in answer.lower():
                    st.markdown(
                        f"""
                        <div class="not-found-card">
                            <h4 style="color: #ef4444; margin-top: 0; margin-bottom: 10px;">❌ Answer Not In Documents</h4>
                            <p style="font-size: 1.1rem; line-height: 1.6; margin: 0; opacity: 0.95; font-weight: 500;">
                                The information requested is not present in the indexed workspace documents.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="answer-card">
                            <h4 style="color: #818cf8; margin-top: 0; margin-bottom: 12px; font-weight: 700;">💡 Answer</h4>
                            <p style="font-size: 1.18rem; line-height: 1.6; margin: 0; color: {TEXT_COLOR}; font-weight: 500;">
                                {answer}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Highlight and display cited sources
                    if citations:
                        matched_sources = []
                        for citation in citations:
                            cite_source = citation.get("source", "")
                            cite_quote = citation.get("exact_quote", "")
                            
                            best_chunk = None
                            for src in sources:
                                if src["source"].lower() == cite_source.lower():
                                    if cite_quote and cite_quote.lower() in src["text"].lower():
                                        best_chunk = src
                                        break
                                    if not best_chunk:
                                        best_chunk = src
                            
                            if best_chunk:
                                highlighted_text = best_chunk["text"]
                                if cite_quote:
                                    escaped_quote = re.escape(cite_quote)
                                    try:
                                        pattern = re.compile(escaped_quote, re.IGNORECASE)
                                        highlight_text_color = "#1e293b"
                                        highlighted_text = pattern.sub(
                                            lambda m: f"<mark style='background-color: #fef08a; color: {highlight_text_color}; padding: 2px 5px; border-radius: 4px; font-weight: bold;'>{m.group(0)}</mark>", 
                                            best_chunk["text"]
                                        )
                                    except Exception:
                                        pass
                                
                                matched_sources.append({
                                    "source": best_chunk["source"],
                                    "page": best_chunk["page"],
                                    "text": highlighted_text,
                                    "quote": cite_quote
                                })
                        
                        if matched_sources:
                            st.markdown("<h4 style='margin-bottom: 12px;'>🔍 Extracted Source References</h4>", unsafe_allow_html=True)
                            for idx, ms in enumerate(matched_sources):
                                st.markdown(
                                    f"""
                                    <div class="glass-card" style="margin-bottom: 15px; padding: 18px; border-left: 4px solid #a855f7;">
                                        <div style="margin-bottom: 10px;">
                                            <span class="source-badge">📄 {ms['source']}</span>
                                            <span class="page-badge">Page {ms['page']}</span>
                                        </div>
                                        <div style="font-size: 1.02rem; line-height: 1.55; color: {TEXT_COLOR}; font-style: italic;">
                                            "{ms['text']}"
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("Citations were returned, but supporting blocks could not be matched.")
                    else:
                        st.info("No specific citations were returned for this answer.")
            st.write("---")

        # 2. Next display suggested questions (BELOW the answer and search results)
        st.markdown("<p style='font-size: 0.88rem; margin-top: 15px; margin-bottom: 8px; opacity: 0.85; font-weight: 600;'>💡 Suggested Questions (Click to Ask):</p>", unsafe_allow_html=True)
        
        # Load sample questions dynamically based on current indexed files (regenerated if new uploads occur)
        sample_questions = rag.get_suggested_questions(os.getcwd())
        
        sq_cols = st.columns(2)
        for idx, sq in enumerate(sample_questions):
            col_idx = idx % 2
            with sq_cols[col_idx]:
                if st.button(sq, key=f"sq_btn_{idx}", use_container_width=True):
                    st.session_state.selected_sample_query = sq
                    st.rerun()
