import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load embedding model (ONCE)
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Embedding function
# -------------------------------
def create_embedding(text_list):
    """
    Converts a list of text chunks into embeddings
    using sentence-transformers.
    """
    embeddings = embedder.encode(
        text_list,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings


# -------------------------------
# Build model from extracted text
# -------------------------------
def build_model(text, model_name):
    os.makedirs("models", exist_ok=True)

    # -------- Chunk text --------
    words = text.split()
    chunks = [
        " ".join(words[i:i + 300]).lower()
        for i in range(0, len(words), 300)
    ]

    # -------- Create embeddings --------
    embeddings = create_embedding(chunks)

    # -------- Store in DataFrame --------
    records = []
    for i, chunk_text in enumerate(chunks):
        records.append({
            "chunk_id": i,
            "text": chunk_text,
            "embedding": embeddings[i]
        })

    df = pd.DataFrame(records)

    # -------- Save model --------
    model_path = f"models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(df, f)

    return model_path
