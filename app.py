import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# --- Importa tu lógica RAG + Ricky ---
from rag_llm import query_qdrant, send_user_message, history

# --- Setup ---
load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=url, api_key=api_key)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection_name = "emailclassification_test"

st.set_page_config(page_title="Email Classifier Dashboard", layout="wide")

# --- Navegación ---
st.sidebar.title("🧭 Navegación")
page = st.sidebar.radio("Ir a:", ["📊 Dashboard", "💬 Asistente RAG"])

# ====================================================
# 📊 DASHBOARD CLÁSICO
# ====================================================
if page == "📊 Dashboard":
    st.title("📬 Email Classification & Semantic Search")

    # --- Sidebar filters ---
    st.sidebar.header("Filtros de búsqueda")
    search_text = st.sidebar.text_input("Buscar por texto (semántico)", "")
    selected_label = st.sidebar.selectbox(
        "Filtrar por etiqueta",
        options=["All", "queja", "petición de servicio", "sugerencia de mejora"]
    )

    # --- Load CSV ---
    df = pd.read_csv("emails_clasificados.csv")

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

        st.subheader("🔍 Resultados de búsqueda semántica")
        for r in results:
            st.markdown(f"**Score:** {r.score:.3f}")
            st.write(r.payload["text"])
            st.caption(f"Etiqueta: {r.payload['label']} | Confianza: {r.payload['confidence']:.2f}")
            st.divider()

    # --- Main table ---
    st.subheader("📊 Todos los emails")
    st.dataframe(df[["etiqueta_predicha", "confianza", "Descripción"]])

# ====================================================
# 💬 ASISTENTE RAG (Ricky Gervais)
# ====================================================
elif page == "💬 Asistente RAG":
    st.title("💬 Asistente RAG (Ricky Gervais Edition)")
    st.caption("Haz preguntas sobre los correos o el dataset. Ricky te responderá con sarcasmo y contexto.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Escribe tu pregunta aquí...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        # Recuperar contexto desde Qdrant
        with st.spinner("Buscando mails relevantes..."):
            context = query_qdrant(user_input)

        formatted_context = "\n\n".join(
            [f"Mail {i+1}:\n{chunk}" for i, chunk in enumerate(context)]
        )

        with st.expander("📬 Contexto recuperado"):
            st.markdown(formatted_context if formatted_context else "Sin contexto relevante.")

        # Capturar respuesta del modelo (redirigiendo el print)
        import io, sys
        buffer = io.StringIO()
        sys.stdout = buffer
        send_user_message(user_input, retrieved_context=context)
        sys.stdout = sys.__stdout__
        model_response = buffer.getvalue().strip()

        with st.chat_message("assistant"):
            st.markdown(model_response)

        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": model_response
        })

    if st.session_state.chat_history:
        with st.expander("🕰️ Historial de conversación"):
            for msg in st.session_state.chat_history:
                st.markdown(f"**Tú:** {msg['user']}")
                st.markdown(f"**Ricky:** {msg['assistant']}")
                st.markdown("---")
