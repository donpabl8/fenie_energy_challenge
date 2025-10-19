import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# --- Setup ---
load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=url, api_key=api_key)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection_name = "emailclassification_test"

st.set_page_config(page_title="Email Classifier Dashboard", layout="wide")
st.title("📬 Email Classification & Semantic Search")

# --- Sidebar controls ---
st.sidebar.header("Search / Filter")
search_text = st.sidebar.text_input("Search by text (semantic)", "")
selected_label = st.sidebar.selectbox(
    "Filter by predicted label",
    options=["All", "queja", "petición de servicio", "sugerencia de mejora"]
)

# --- Load local CSV for table display ---
df = pd.read_csv("emails_clasificados.csv")

# print(df.columns)

if selected_label != "All":
    df = df[df["etiqueta_predicha"] == selected_label]

# --- Semantic search section ---
if search_text:
    query_vector = model.encode(search_text).tolist()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        with_payload=True
    )
    st.subheader("🔍 Semantic Search Results")
    for r in results:
        st.markdown(f"**Score:** {r.score:.3f}")
        st.write(r.payload["text"])
        st.caption(f"Label: {r.payload['label']} | Confidence: {r.payload['confidence']:.2f}")
        st.divider()

# --- Main table ---
st.subheader("📊 All Emails")
st.dataframe(df[["etiqueta_predicha", "confianza", "Descripción"]])
