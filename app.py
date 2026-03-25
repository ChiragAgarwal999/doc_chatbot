import streamlit as st
import os
import time
import pickle
import requests
import numpy as np
from dotenv import load_dotenv
from extractor import extract_text
from model_builder import build_model, create_embedding
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

if "last_selected_model" not in st.session_state:
    st.session_state.last_selected_model = "— Select —"


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📄",
    layout="wide"
)

# -------------------------------------------------
# App Title & Description
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>📄 Document Chatbot</h1>
    <p style='text-align: center; color: gray;'>
        Upload a document, build a temporary model, and ask questions from it .
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Sidebar - Model Management
# -------------------------------------------------
st.sidebar.header("📁 Document Models")

os.makedirs("models", exist_ok=True)
model_files = sorted(os.listdir("models"))

if not model_files:
    st.sidebar.info("No models available yet.")

selected_model = st.sidebar.selectbox(
    "Select a document model",
    ["— Select —"] + model_files
)

if selected_model != st.session_state.last_selected_model:
    st.session_state.query_text = ""   # clear question input
    st.session_state.last_selected_model = selected_model


st.sidebar.markdown("---")
st.sidebar.caption(
    "Models are stored temporarily.\n"
    "They can later be migrated to cloud or database."
)

# -------------------------------------------------
# File Upload Section
# -------------------------------------------------
st.subheader("📤 Upload Document")

uploaded_file = st.file_uploader(
    "Supported formats: PDF, DOCX, TXT",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("⚙️ Create Model", use_container_width=True):
            with st.spinner("Extracting text and building model..."):
                text = extract_text(uploaded_file)

                # 🚨 Detect image-based PDF
                if not text or len(text.strip()) < 50:
                    st.warning(
                        "⚠️ This document appears to be an image-based PDF.\n\n"
                        "Please upload a **text-based PDF, DOCX, or TXT file**.\n\n"
                        "Scanned documents are not supported yet."
                    )
                    st.stop()

                model_name = uploaded_file.name.replace(".", "_")
                build_model(text, model_name)

            st.success("✅ Model created successfully!")
            time.sleep(0.5)
            st.rerun()


    with col2:
        st.info(
            f"**File:** {uploaded_file.name}\n\n"
            f"**Size:** {round(uploaded_file.size / 1024, 2)} KB"
        )

st.markdown("---")

# -------------------------------------------------
# Chat Section
# -------------------------------------------------
st.subheader("💬 Ask Questions")

if selected_model != "— Select —":
    model_path = f"models/{selected_model}"

    with open(model_path, "rb") as f:
        df = pickle.load(f)

    query = st.text_input(
        "Enter your question",
    placeholder="e.g. What is the main topic discussed in this document?",
    key="query_text"
)


    if query:
        with st.spinner("Searching relevant context..."):
            # -------- Embed query --------
            question_embedding = create_embedding([query.lower()])[0]

            similarities = cosine_similarity(
                np.vstack(df["embedding"]),
                [question_embedding]
            ).flatten()

            top_k = 3
            top_idx = similarities.argsort()[::-1][:top_k]
            context_chunks = df.iloc[top_idx]["text"].tolist()

        # -------- Display retrieved chunks --------
        st.markdown("### 🧠 Retrieved Context")
        for i, chunk in enumerate(context_chunks, 1):
            with st.expander(f"Context {i}"):
                st.write(chunk)

        # -------- Build prompt for Gemini --------
        combined_context = "\n\n".join(
            [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )

        prompt = f"""
You are an expert AI assistant.

Your task is to answer the user's question using ONLY the information provided in the document context below.
Do NOT use external knowledge or assumptions.

Answer Guidelines:
- Be clear, concise, and accurate
- Use bullet points wherever possible
- Use short headings if they improve clarity
- Avoid repetition
- Do not add information not present in the context

If the answer cannot be found in the context, reply exactly with:
"I don’t know based on the provided document."

====================
DOCUMENT CONTEXT:
{combined_context}
====================

User Question:
{query}

Structured Answer:
"""


                # -------------------------- 
        # Gemini API Call (UPDATED)
        # --------------------------
        with st.spinner("Generating answer..."):
            bot_placeholder = st.empty()
            try:
                response = client.models.generate_content(
                    model="models/gemini-flash-latest",  # ✅ correct format
                    contents=prompt,
                    config={
                        "temperature": 0.3,
                        "max_output_tokens": 1024,
                    }
                )

                final_answer = response.text.strip()

            except Exception as e:
                final_answer = f"⚠️ Gemini API Error: {e}"


        # --------------------------
        # Typing Effect Simulation
        # --------------------------
        typed_text = ""
        for char in final_answer:
            typed_text += char
            bot_placeholder.markdown(
                f"<div class='stChatMessage bot-msg'>{typed_text}▋</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.02)

        bot_placeholder.markdown(
            f"<div class='stChatMessage bot-msg'>{final_answer}</div>",
            unsafe_allow_html=True
        )

else:
    st.info("Please select a document model from the sidebar to start chatting.")
