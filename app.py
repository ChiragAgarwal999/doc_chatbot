import streamlit as st
import os
import time
import pickle
import re
import hashlib
import html
import base64
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
from extractor import extract_text
from model_builder import build_model, create_embedding
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
LOGO_ICON_URL = "https://img.icons8.com/?size=100&id=vrSyrcgYoUsG&format=png&color=000000"
ASK_ICON_URL = "https://img.icons8.com/?size=100&id=67444&format=png&color=000000"

# Prefer higher daily quota models first, then fall back automatically.
MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None

if "processed_upload_hashes" not in st.session_state:
    st.session_state.processed_upload_hashes = set()

if "doc_frames" not in st.session_state:
    st.session_state.doc_frames = {}

if "processing_notice" not in st.session_state:
    st.session_state.processing_notice = ""

if "last_seen_upload_hash" not in st.session_state:
    st.session_state.last_seen_upload_hash = None


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI Doc Chatbot",
    page_icon="📄",
    layout="wide"
)


def _safe_model_name(filename, content_hash):
    stem = os.path.splitext(filename)[0].lower().strip()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    stem = stem or "document"
    return f"{stem}_{content_hash[:8]}"


def _get_doc_by_id(doc_id):
    for doc in st.session_state.documents:
        if doc["id"] == doc_id:
            return doc
    return None


def _load_doc_df(model_path):
    if model_path in st.session_state.doc_frames:
        return st.session_state.doc_frames[model_path]

    with open(model_path, "rb") as f:
        frame = pickle.load(f)

    st.session_state.doc_frames[model_path] = frame
    return frame


def _display_name_from_model_path(model_path):
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    # Model names are saved as <safe_file_name>_<hash8>. Keep a readable fallback.
    readable_name = re.sub(r"_[0-9a-f]{8}$", "", base_name)
    readable_name = readable_name.replace("_", " ").strip()
    return readable_name or base_name


def _sync_documents_from_models():
    os.makedirs("models", exist_ok=True)

    model_paths = []
    for file_name in sorted(os.listdir("models")):
        if file_name.lower().endswith(".pkl"):
            model_paths.append(os.path.join("models", file_name))

    # Drop documents that no longer exist on disk.
    valid_docs = [
        doc for doc in st.session_state.documents
        if os.path.exists(doc.get("model_path", ""))
    ]
    st.session_state.documents = valid_docs

    known_paths = {doc["model_path"] for doc in st.session_state.documents}

    # Register models already present on disk (important after full page refresh/restart).
    for model_path in model_paths:
        if model_path in known_paths:
            continue

        stat = os.stat(model_path)
        doc_id = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:12]
        doc_record = {
            "id": doc_id,
            "hash": None,
            "name": _display_name_from_model_path(model_path),
            "size_kb": round(stat.st_size / 1024, 2),
            "model_path": model_path,
            "created_at": stat.st_mtime,
        }

        st.session_state.documents.append(doc_record)
        st.session_state.chat_histories.setdefault(doc_id, [])

    st.session_state.documents.sort(key=lambda d: d.get("created_at", 0), reverse=True)

    if st.session_state.active_doc_id is not None:
        active_exists = any(
            doc["id"] == st.session_state.active_doc_id
            for doc in st.session_state.documents
        )
        if not active_exists:
            st.session_state.active_doc_id = None

    if st.session_state.active_doc_id is None and st.session_state.documents:
        st.session_state.active_doc_id = st.session_state.documents[0]["id"]


def _register_document(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    if content_hash in st.session_state.processed_upload_hashes:
        for doc in st.session_state.documents:
            if doc["hash"] == content_hash:
                st.session_state.active_doc_id = doc["id"]
                st.session_state.processing_notice = "This file is already indexed. Opened existing chat."
                return

    file_like = BytesIO(file_bytes)
    file_like.name = uploaded_file.name

    text = extract_text(file_like)

    if not text or len(text.strip()) < 50:
        st.sidebar.warning(
            "This document appears to be image-based or has too little extractable text. "
            "Please upload a text-based PDF, DOCX, or TXT file."
        )
        return

    model_name = _safe_model_name(uploaded_file.name, content_hash)

    model_path = build_model(text, model_name)

    doc_id = content_hash[:12]
    doc_record = {
        "id": doc_id,
        "hash": content_hash,
        "name": uploaded_file.name,
        "size_kb": round(uploaded_file.size / 1024, 2),
        "model_path": model_path,
        "created_at": time.time()
    }

    st.session_state.documents.insert(0, doc_record)
    st.session_state.processed_upload_hashes.add(content_hash)
    st.session_state.chat_histories[doc_id] = [
        {
            "role": "assistant",
            "content": "Document indexed successfully. Ask me anything from this file.",
            "contexts": []
        }
    ]
    st.session_state.active_doc_id = doc_id
    st.session_state.processing_notice = "Document processed and ready to chat."


def _render_retrieved_context(contexts, key_prefix):
    if not contexts:
        return

    chip_items = []
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", key_prefix)

    for idx, chunk in enumerate(contexts, 1):
        safe_chunk = html.escape(str(chunk)).replace("\n", "<br>")
        chip_items.append(
            (
                f'<details class="context-chip-wrap" name="ctx_group_{safe_prefix}">'
                f'<summary class="context-chip" aria-label="Show context {idx}">Context {idx}</summary>'
                f'<div class="context-tooltip">'
                f'<div class="context-tooltip-title">Chunk {idx}</div>'
                f'<div class="context-tooltip-body">{safe_chunk}</div>'
                f'</div>'
                f'</details>'
            )
        )

    st.markdown(
        f"<div class='context-chip-row' id='ctx_{safe_prefix}'>{''.join(chip_items)}</div>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# App Title & Description
# -------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg-a: #f8fff6;
        --bg-b: #ecf5ff;
        --brand: #136f63;
        --brand-soft: #d8efe8;
        --text-dark: #0f1419;
        --text-body: #2d3748;
        --border: #cbd5e0;
        --accent: #0066cc;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, #dff8f0 0%, transparent 28%),
            radial-gradient(circle at 100% 100%, #dce8ff 0%, transparent 30%),
            linear-gradient(140deg, var(--bg-a) 0%, var(--bg-b) 100%);
        overflow-x: clip;
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        max-width: 100%;
        overflow-x: clip !important;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    header {
        background: #2f3440 !important;
        border-bottom: 1px solid #d7e2ee !important;
    }

    /* General text color overrides */
    h1, h2, h3, p, [data-testid="stMarkdownContainer"] {
        color: var(--text-body);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Chat messages styling */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    .chat-message {
        padding: 12px 14px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    .user-msg {
        background: rgba(0, 102, 204, 0.08);
        border-left: 3px solid var(--accent);
    }

    .assistant-msg {
        background: rgba(255, 255, 255, 0.6);
        border-left: 3px solid #10b981;
    }

    .hero {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 12px;
        backdrop-filter: blur(4px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .project-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        margin: 0;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #cfd8e6;
        color: #13263d;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.98rem;
        letter-spacing: 0.01em;
        position: fixed;
        top: 82px;
        left: calc(var(--sidebar-width) + 24px);
        z-index: 9997;
        box-shadow: 0 10px 24px rgba(12, 23, 40, 0.2), 0 2px 6px rgba(12, 23, 40, 0.16);
    }

    .about-team-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 14px;
        border-radius: 999px;
        background: #13263d;
        color: #f4f8ff !important;
        border: 1px solid #3f5977;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none !important;
        position: fixed;
        top: 82px;
        right: 24px;
        z-index: 9997;
        box-shadow: 0 10px 24px rgba(12, 23, 40, 0.2), 0 2px 6px rgba(12, 23, 40, 0.16);
        transition: transform 0.15s ease, background 0.2s ease;
    }

    .about-team-btn:hover {
        background: #1c3655;
        transform: translateY(-1px);
    }

    .about-overlay {
        position: fixed;
        inset: 0;
        z-index: 20000;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(8, 15, 28, 0.56);
        backdrop-filter: blur(12px) saturate(118%);
        -webkit-backdrop-filter: blur(12px) saturate(118%);
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
        transition: opacity 0.28s ease, visibility 0.28s ease;
    }

    .about-overlay:target {
        opacity: 1;
        visibility: visible;
        pointer-events: auto;
    }

    .about-card {
        position: relative;
        width: 100%;
        border: none;
        background: transparent;
        box-shadow: none;
        border-radius: 28px;
        padding: 20px;
        max-height: calc(100vh - 56px);
        overflow: hidden;
        transform: translateY(18px) scale(0.985);
        transition: transform 0.3s ease;
    }

    .about-overlay:target .about-card {
        transform: translateY(0) scale(1);
    }

    .about-close {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        text-decoration: none !important;
        background: rgba(242, 247, 255, 0.12);
        border: 1px solid rgba(242, 247, 255, 0.24);
        color: #f5f8ff !important;
        font-size: 1.1rem;
        line-height: 1;
        position: absolute;
        top: 14px;
        right: 14px;
        z-index: 20010;
    }

    .about-layout {
        display: grid;
        grid-template-columns: 1.05fr 0.95fr;
        gap: 20px;
        align-items: stretch;
        max-height: calc(100vh - 96px);
    }

    .about-copy {
        padding: 12px 6px 12px 4px;
        color: #e8eef8;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        padding-right: 10px;
    }

    .about-copy::-webkit-scrollbar,
    .about-visual::-webkit-scrollbar {
        width: 8px;
    }

    .about-copy::-webkit-scrollbar-thumb,
    .about-visual::-webkit-scrollbar-thumb {
        background: rgba(200, 214, 235, 0.45);
        border-radius: 999px;
    }

    .about-copy::-webkit-scrollbar-track,
    .about-visual::-webkit-scrollbar-track {
        background: transparent;
    }

    .about-visual {
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        padding-right: 6px;
    }

    .about-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: #b5c8e6;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .about-copy h2 {
        margin: 12px 0 8px;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        line-height: 1.12;
    }

    .about-copy p {
        margin: 0 0 14px;
        color: #d8e2f2;
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .about-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }

    .about-pill {
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #edf4ff;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .about-list {
        display: grid;
        gap: 10px;
        margin: 0;
    }

    .about-list-item {
        padding: 12px 14px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .about-list-item strong {
        display: block;
        margin-bottom: 4px;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
    }

    .about-list-item span {
        color: #c7d5e8;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .team-members {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 6px;
    }

    .team-member-card {
        padding: 14px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .team-member-card:nth-child(1) {
        border-left: 4px solid #ffb648;
    }

    .team-member-card:nth-child(2) {
        border-left: 4px solid #6ad7ff;
    }

    .team-member-card:nth-child(3) {
        border-left: 4px solid #8ef0b6;
    }

    .team-member-card:nth-child(4) {
        border-left: 4px solid #c9a7ff;
    }

    .team-member-role {
        display: inline-flex;
        align-items: center;
        margin-bottom: 8px;
        padding: 4px 8px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: #edf4ff;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .team-member-name {
        display: block;
        margin-bottom: 6px;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.98rem;
        font-weight: 700;
    }

    .team-member-desc {
        color: #c7d5e8;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .about-image-wrap {
        border-radius: 22px;
        overflow: hidden;
        border: none;
        background: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 360px;
        padding: 18px;
    }

    .about-image-wrap img {
        width: 100%;
        max-width: 100%;
        max-height: 70vh;
        object-fit: contain;
        display: block;
        filter: drop-shadow(0px -10px 19px orange);
    }

    .about-image-fallback {
        color: #d9e6fb;
        font-size: 0.95rem;
        text-align: center;
        padding: 26px 18px;
    }

    .about-team-name {
        margin-top: 14px;
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #ffb648;
        text-transform: uppercase;
        filter: none;
    }

    .about-team-subtitle {
        margin-top: 6px;
        text-align: center;
        color: #c7d5e8;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .project-logo img,
    .sidebar-title img {
        width: 22px;
        height: 22px;
        object-fit: contain;
        vertical-align: middle;
    }

    .sidebar-title {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .ask-title {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .ask-title img {
        width: 28px;
        height: 28px;
        object-fit: contain;
    }

    .hero h1 {
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        letter-spacing: -0.02em;
        color: var(--text-dark);
    }

    .hero p {
        margin: 6px 0 0 0;
        color: var(--text-body);
        opacity: 0.9;
    }

    .doc-card {
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.75);
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: var(--text-dark);
        font-weight: 500;
    }

    .small-note {
        font-size: 0.86rem;
        color: var(--text-body);
        opacity: 0.75;
    }

    .stMetric,
    .stInfo,
    .stWarning,
    .stError,
    .stSuccess {
        border-radius: 12px;
    }

    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #1e2139 0%, #242d47 100%) !important;
        position: relative;
    }

    .stSidebar [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    .stSidebar h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin-bottom: 16px !important;
    }

    .stSidebar p {
        color: #e0e0e0 !important;
    }

    .stSidebar caption {
        color: #a0a0a0 !important;
        font-size: 0.8rem !important;
    }

    .stSidebar .stButton > button {
        background: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s !important;
    }

    .stSidebar .stButton > button:hover {
        background: rgba(0, 102, 204, 0.3) !important;
        border-color: #0066cc !important;
    }

    .stSidebar hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Retrieved context chips (single row with wrap + hover preview) */
    .context-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 12px;
        align-items: center;
    }

    .context-chip-wrap {
        position: relative;
        display: inline-block;
    }

    .context-chip-wrap > summary {
        list-style: none;
    }

    .context-chip-wrap > summary::-webkit-details-marker {
        display: none;
    }

    .context-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 30px;
        padding: 4px 12px;
        border-radius: 999px;
        border: 1px solid #b9cadf;
        background: linear-gradient(180deg, #f6faff 0%, #eaf2fc 100%);
        color: #1a4b77;
        font-size: 0.87rem;
        font-weight: 600;
        cursor: pointer;
        user-select: none;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        outline: none;
    }

    .context-chip:hover,
    .context-chip:focus-visible {
        border-color: #4f88bf;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(31, 77, 124, 0.18);
        outline: none;
    }

    .context-tooltip {
        position: absolute;
        bottom: calc(100% + 10px);
        left: 0;
        width: min(560px, 86vw);
        background: #ffffff;
        border: 1px solid #c8d7e8;
        border-radius: 12px;
        box-shadow: 0 14px 28px rgba(14, 35, 59, 0.2);
        padding: 10px 12px;
        z-index: 2000;
        opacity: 0;
        visibility: hidden;
        transform: translateY(6px);
        transition: opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;
        pointer-events: auto;
    }

    .context-tooltip-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: #234b77;
        margin-bottom: 6px;
    }

    .context-tooltip-body {
        max-height: 210px;
        overflow-y: auto;
        color: #243b53;
        font-size: 0.84rem;
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .context-chip-wrap[open] .context-tooltip {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }

    .context-chip-wrap[open] .context-chip {
        border-color: #4f88bf;
        box-shadow: 0 6px 16px rgba(31, 77, 124, 0.18);
    }

    .sidebar-doc-size {
        font-size: 0.75rem;
        color: #a0a0a0;
        margin-top: 4px;
    }

    /* Bottom chat input styling */
    [data-testid="stChatInputContainer"] {
        background: #ecf5fd !important;
        border-top: none !important;
    }

    [data-testid="stChatInputContainer"] > div {
        background: #ecf5fd !important;
    }

    [data-testid="stBottomBlockContainer"] {
        background: #ecf5fd !important;
    }

    [data-testid="stChatInput"] {
        background: transparent !important;
    }

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form {
    display: flex !important;
    flex-flow: row-reverse nowrap !important;
    background: #2f3440 !important;
    border: 1px solid #4b5568 !important;
    border-radius: 999px !important;
    box-shadow: 0 8px 20px rgba(15, 35, 65, 0.18) !important;

    min-height: 60px !important;
    max-height: 60px !important;
    padding: 8px 12px !important;
    gap: 8px !important;
    overflow: hidden !important;
    align-items: center !important;
}

    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div,
    [data-testid="stChatInput"] [data-baseweb="input"] > div {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        background: #2f3440 !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: #2f3440 !important;
        border: none !important;
        color: #f1f5fb !important;
        outline: none !important;
        box-shadow: none !important;
        -webkit-tap-highlight-color: transparent;
    }

[data-testid="stChatInput"] textarea {
    height: auto !important;
    min-height: 24px !important;
    max-height: 140px !important;

    overflow-y: auto !important;
    overflow-x: hidden !important;

    resize: none !important;

    /* 🔥 KEY FIX */
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;

    line-height: 1.4 !important;
    padding: 6px 0 !important;
}

[data-testid="stChatInput"] {
    width: 100% !important;
    max-width: 100% !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form {
    width: 100% !important;
}

    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] input:focus,
    [data-testid="stChatInput"] textarea:focus-visible,
    [data-testid="stChatInput"] input:focus-visible,
    [data-testid="stChatInput"] form:focus-within,
    [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
    [data-testid="stChatInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div:focus-within,
    [data-testid="stChatInput"] [data-baseweb="input"] > div:focus-within {
        outline: none !important;
        box-shadow: none !important;
        border-color: #4b5568 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #b5becd !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] button {
        position: static !important;
        inset: auto !important;
        transform: none !important;
        align-self: center !important;
        margin: 0 !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #f1f5fb !important;
        border: 1px solid #d0d9e6 !important;
        border-radius: 50% !important;
        color: #2c3e50 !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #e3ebf6 !important;
        border-color: #bfcddd !important;
    }

    [data-testid="stStatusWidget"],
    section[data-testid="stStatusWidget"] {
        background: #000000 !important;
        border: 1px solid #2b2b2b !important;
        border-radius: 10px !important;
        margin: 2px 0 8px 0 !important;
    }

    [data-testid="stStatusWidget"] > div,
    [data-testid="stStatusWidget"] > div > div,
    [data-testid="stStatusWidget"] div[role="status"] {
        background: #000000 !important;
    }

    [data-testid="stStatusWidget"] details,
    [data-testid="stStatusWidget"] details:hover,
    [data-testid="stStatusWidget"] details:focus-within,
    [data-testid="stStatusWidget"] details[open],
    [data-testid="stStatusWidget"] summary,
    [data-testid="stStatusWidget"] summary:hover,
    [data-testid="stStatusWidget"] summary:focus,
    [data-testid="stStatusWidget"] summary:focus-visible,
    [data-testid="stStatusWidget"] details[open] > summary {
        background: #000000 !important;
        border-radius: 10px !important;
    }

    [data-testid="stStatusWidget"] summary {
        min-height: 36px !important;
        padding: 6px 10px !important;
        transition: none !important;
        box-shadow: none !important;
        border: none !important;
    }

    [data-testid="stStatusWidget"] summary > div,
    [data-testid="stStatusWidget"] details > div,
    [data-testid="stStatusWidget"] [data-testid="stVerticalBlock"] {
        background: #000000 !important;
    }

    /* Some Streamlit versions render status rows as expanders. Keep them black in all states. */
    [data-testid="stExpander"],
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:focus-visible,
    [data-testid="stExpander"] details[open] > summary,
    [data-testid="stExpander"] summary > div,
    [data-testid="stExpander"] summary > div > div {
        background: #000000 !important;
        color: #ffffff !important;
        border-color: #2b2b2b !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] label {
        color: #ffffff !important;
    }

    [data-testid="stExpander"] svg,
    [data-testid="stExpander"] svg * {
        color: #ffffff !important;
        stroke: #ffffff !important;
    }

    [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    [data-testid="stStatusWidget"] p,
    [data-testid="stStatusWidget"] span {
        color: #ffffff !important;
    }

    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] svg * {
        color: #ffffff !important;
        stroke: #ffffff !important;
    }

    .retrieved-context-title {
        margin: 6px 0 6px 0;
        color: #1f3550;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .page-footer {
        position: fixed;
        left: 0;
        bottom: 12px;
        width: 100%;
        z-index: 9998;
        display: flex;
        justify-content: center;
        background: transparent;
        color: #000000;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        border: none;
        font-size: 0.82rem;
        line-height: 1;
        white-space: nowrap;
        pointer-events: none;
        text-shadow: none;
    }

    .page-footer strong {
        color: #000000;
        font-weight: 600;
        text-shadow: none;
    }

    @media (max-width: 768px) {
        [data-testid="block-container"] {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] form {
            min-height: 56px !important;
            max-height: 56px !important;
        }

        [data-testid="stChatInput"] textarea {
                height: 20px !important;
                min-height: 20px !important;
                max-height: 20px !important;
        }

        .project-logo {
            position: static;
            left: auto;
            top: auto;
            width: fit-content;
            max-width: 100%;
            margin-bottom: 8px;
        }

        .about-team-btn {
            position: static;
            top: auto;
            right: auto;
            width: fit-content;
            max-width: 100%;
            margin: 0 0 8px 0;
        }

        .about-overlay {
            padding: 12px;
        }

        .about-close {
            top: 12px;
            right: 12px;
        }

        .about-image-wrap img {
            max-width: 94vw;
            max-height: 82vh;
        }

        .hero {
            padding: 14px 12px;
        }

        .about-card {
            padding: 16px;
            max-height: calc(100vh - 24px);
            overflow-y: auto;
        }

        .about-layout,
        .about-copy,
        .about-visual {
            max-height: none;
            overflow: visible;
            padding-right: 0;
        }

        .about-layout {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .about-visual {
            order: -1;
        }

        .about-copy {
            order: 1;
        }

        .about-copy h2 {
            font-size: 1.55rem;
        }

        .about-list {
            gap: 8px;
        }

        .team-members {
            grid-template-columns: 1fr;
            gap: 10px;
        }

        .team-member-card {
            padding: 12px;
        }

        .about-image-wrap {
            min-height: 240px;
            padding: 10px;
        }

        .about-image-wrap img {
            max-width: 100%;
            max-height: 42vh;
        }

        .about-team-name {
            font-size: 1.08rem;
        }

        .about-team-subtitle {
            font-size: 0.84rem;
        }

        .context-panel-body {
            max-height: 220px;
        }

        .context-chip-row {
            align-items: flex-start;
        }

        .context-chip-wrap {
            display: block;
            width: 100%;
        }

        .context-chip {
            max-width: 100%;
        }

        .context-tooltip {
            position: static;
            left: auto;
            bottom: auto;
            transform: none;
            width: 100%;
            margin-top: 8px;
            opacity: 1;
            visibility: visible;
            transition: none;
        }

        .context-chip-wrap:not([open]) .context-tooltip {
            display: none;
        }

        .context-chip-wrap[open] .context-tooltip {
            transform: none;
        }

        .page-footer {
            position: static;
            width: 100%;
            white-space: normal;
            text-align: center;
            line-height: 1.3;
            padding: 6px 10px 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar - Document History
# -------------------------------------------------
st.sidebar.markdown(
    f"<h2 style='margin-top: 0;'><span class='sidebar-title'><img src='{LOGO_ICON_URL}' alt='doc icon'/>Documents</span></h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<hr style='border-color: rgba(255,255,255,0.2); margin: 12px 0;'>",
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt"],
    key="doc_uploader",
    help="Upload PDF, DOCX, or TXT files"
)

if uploaded_file is not None:
    current_upload_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if current_upload_hash != st.session_state.last_seen_upload_hash:
        _register_document(uploaded_file)
        st.session_state.last_seen_upload_hash = current_upload_hash
else:
    st.session_state.last_seen_upload_hash = None

if st.session_state.processing_notice:
    st.sidebar.success(st.session_state.processing_notice)
    st.session_state.processing_notice = ""

os.makedirs("models", exist_ok=True)
_sync_documents_from_models()

if not st.session_state.documents:
    st.sidebar.info("📂 No documents yet. Upload one to begin.")
else:
    st.sidebar.markdown(
        "<p style='color:#a0a0a0; font-size: 0.85rem; margin-bottom: 10px;'>Your Files</p>",
        unsafe_allow_html=True,
    )
    for doc in st.session_state.documents:
        is_active = st.session_state.active_doc_id == doc["id"]
        button_label = f"📄 {doc['name'][:28]}" if len(doc['name']) > 28 else f"📄 {doc['name']}"
        if st.sidebar.button(
            button_label,
            key=f"doc_{doc['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_doc_id = doc["id"]

        st.sidebar.markdown(
            f"<div class='sidebar-doc-size'>{doc['size_kb']} KB</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
    <div class='project-logo'><img src='{LOGO_ICON_URL}' alt='logo'/>AI Doc Chatbot</div>
    <a class='about-team-btn' href='#about-team-overlay'>About Team</a>
    """,
    unsafe_allow_html=True,
)

active_doc = _get_doc_by_id(st.session_state.active_doc_id)

st.markdown(
    f"""
    <div class='hero'>
        <h1>Document Chat</h1>
        <p>Upload in the sidebar. We index automatically, then you can chat immediately.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not active_doc:
    st.markdown(
        f"""
        <hr style='border-color: rgba(20, 35, 60, 0.18); margin: 16px 0 18px 0;'>
        <h2 style='margin: 0 0 6px 0; color: #16283f;'><span class='ask-title'><img src='{ASK_ICON_URL}' alt='ask icon'/>Ask Questions</span></h2>
        <p style='margin: 0 0 10px 0; font-size: 0.92rem; color: #4a5a6d;'>Enter your question</p>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload a document from the sidebar to start chatting.")
    st.chat_input("Upload a document first to start chatting", disabled=True)
else:
    st.markdown(
        f"""
        <div class='doc-card'><strong>Active Document:</strong> {active_doc['name']}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <hr style='border-color: rgba(20, 35, 60, 0.18); margin: 16px 0 18px 0;'>
        <h2 style='margin: 0 0 6px 0; color: #16283f;'><span class='ask-title'><img src='{ASK_ICON_URL}' alt='ask icon'/>Ask Questions</span></h2>
        <p style='margin: 0 0 10px 0; font-size: 0.92rem; color: #4a5a6d;'>Enter your question</p>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.chat_histories.setdefault(active_doc["id"], [])

    for msg_idx, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("contexts"):
                st.markdown("<div class='retrieved-context-title'>Retrieved Context</div>", unsafe_allow_html=True)
                _render_retrieved_context(
                    msg["contexts"],
                    key_prefix=f"history_{active_doc['id']}_{msg_idx}",
                )
            st.markdown(msg["content"])

    st.caption("Use the bottom box to ask your question.")
    user_question = st.chat_input("Ask a question about this document")

    if user_question:
        history.append({"role": "user", "content": user_question, "contexts": []})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            progress = st.status("Thinking", expanded=False)

            df = _load_doc_df(active_doc["model_path"])
            question_embedding = create_embedding([user_question.lower()])[0]

            similarities = cosine_similarity(
                np.vstack(df["embedding"]),
                [question_embedding]
            ).flatten()

            top_k = 5
            top_idx = similarities.argsort()[::-1][:top_k]
            context_chunks = df.iloc[top_idx]["text"].tolist()

            # Show retrieved context first, then generate answer (same as previous UI behavior).
            progress.update(label="Retrieving relevant context", state="running", expanded=False)
            st.markdown("<div class='retrieved-context-title'>Retrieved Context</div>", unsafe_allow_html=True)
            _render_retrieved_context(
                context_chunks,
                key_prefix=f"live_{active_doc['id']}_{len(history)}",
            )

            progress.update(label="Generating answer", state="running", expanded=False)

            combined_context = "\n\n".join(
                [f"Context {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
            )

            prompt = f"""
You are an expert AI assistant.

Your task is to answer the user's question using ONLY the information provided in the document context below.
Do NOT use external knowledge or assumptions.

Answer Guidelines:
- Be accurate, insightful, and well-explained
- Prefer a polished answer with short section headings
- Use bullets for lists, but write natural explanatory sentences when needed
- Include concrete details from the context (names, skills, numbers, timelines) where relevant
- Keep it readable and engaging, not robotic
- Do not add information not present in the context

If the answer cannot be found in the context, reply exactly with:
"I don't know based on the provided document."

====================
DOCUMENT CONTEXT:
{combined_context}
====================

User Question:
{user_question}

Structured Answer:
"""

            placeholder = st.empty()
            
            max_retries = 3
            retry_delay = 2
            final_answer = None

            for model_name in MODEL_CANDIDATES:
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            temperature=0.55,
                            max_tokens=1400,
                        )
                        final_answer = (response.choices[0].message.content or "").strip()
                        break
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "rate" in error_str.lower():
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2
                        else:
                            final_answer = f"Groq API Error: {e}"
                            break

                if final_answer:
                    break

                # Reset backoff for next model candidate.
                retry_delay = 2

            if final_answer is None:
                final_answer = "🚫 API rate limit exceeded across available models. Please try again later."
            
            if final_answer == "":
                final_answer = "Unexpected empty response. Please try again."

            typed_text = ""
            for char in final_answer:
                typed_text += char
                placeholder.markdown(f"{typed_text}▋")
                time.sleep(0.008)

            placeholder.markdown(final_answer)
            progress.update(label="Answer ready", state="complete", expanded=False)

        history.append(
            {
                "role": "assistant",
                "content": final_answer,
                "contexts": context_chunks,
            }
        )
        st.rerun()

st.markdown(
    "<div id='team-info' class='page-footer'>Develop with 💖 by <strong>Team Sprint Savants</strong></div>",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# About Team Section
# -------------------------------------------------
TEAM_IMAGE_URL = "assets/team-group.png"


def _render_about_team_section():
    image_src = TEAM_IMAGE_URL
    if os.path.exists(TEAM_IMAGE_URL):
        with open(TEAM_IMAGE_URL, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        image_src = f"data:image/png;base64,{encoded}"

    image_block = f"<img src='{image_src}' alt='Team Sprint Savants'/>"

    st.markdown(
        f"""
        <section id='about-team-overlay' class='about-overlay'>
            <div class='about-card'>
                <a class='about-close' href='#' aria-label='Close about section'>✕</a>
                <div class='about-layout'>
                    <div class='about-copy'>
                        <div class='about-eyebrow'>Who built this</div>
                        <h2>Team Sprint Savants</h2>
                        <div class='about-list'>
                            <div class='team-members'>
                                <div class='team-member-card'>
                                    <span class='team-member-role'>Leader</span>
                                    <span class='team-member-name'>Jatin Chutani</span>
                                    <span class='team-member-desc'>Coordinated the team, handled presentation flow, and organized the final PPT and poster delivery.</span>
                                </div>
                                <div class='team-member-card'>
                                    <span class='team-member-role'>UI / UX</span>
                                    <span class='team-member-name'>Manish Jangir</span>
                                    <span class='team-member-desc'>Designed the interface look, refined layout spacing, and improved the chatbot interaction feel.</span>
                                </div>
                                <div class='team-member-card'>
                                    <span class='team-member-role'>Research Support</span>
                                    <span class='team-member-name'>Asha Kanwer</span>
                                    <span class='team-member-desc'>Supported web scraping work and helped prepare the presentation and poster materials.</span>
                                </div>
                                <div class='team-member-card'>
                                    <span class='team-member-role'>Core Development</span>
                                    <span class='team-member-name'>Chirag Agarwal</span>
                                    <span class='team-member-desc'>Built the main backend flow, model integration, and core chatbot functionality end to end.</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class='about-visual'>
                        <div class='about-image-wrap'>
                            {image_block}
                        </div>
                        <div class='about-team-name'>Team Sprint Savants</div>
                        <div class='about-team-subtitle'>A collaborative effort focused on practical AI, clean interface design, and clear project communication from development to presentation.</div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


_render_about_team_section()
