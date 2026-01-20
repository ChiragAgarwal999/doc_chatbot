import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch



from transformers import BertTokenizer, BertModel
# Load Google BERT once globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # set to evaluation mode

def create_embedding(text_list, max_length=512):
    """
    Convert a list of text chunks into embeddings using Google BERT.

    Args:
        text_list (list[str]): List of text chunks.
        max_length (int): Maximum token length for BERT input.

    Returns:
        np.ndarray: Embeddings array of shape (len(text_list), 768)
    """
    embeddings = []

    for text in text_list:
        # Tokenize text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)

    return np.vstack(embeddings)



# # -------------------------------
# # Load embedding model (ONCE)
# # -------------------------------
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # -------------------------------
# # Embedding function
# # -------------------------------
# def create_embedding(text_list):
#     """
#     Converts a list of text chunks into embeddings
#     using sentence-transformers.
#     """
#     embeddings = embedder.encode(
#         text_list,
#         convert_to_numpy=True,
#         show_progress_bar=False
#     )
#     return embeddings


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
