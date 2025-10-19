import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import uuid

from rag_llm import query_qdrant, send_user_message, history

# --- Setup ---
load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=url, api_key=api_key)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection_name = "emailclassification_test"

df = pd.read_csv("emails_clasificados.csv")
df_original = df.copy()
filtered_df = df_original.copy()

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

    if "Fecha" in filtered_df.columns:
        st.sidebar.markdown("### Filtro por fecha")
        filtered_df["Fecha"] = pd.to_datetime(filtered_df["Fecha"], errors="coerce", infer_datetime_format=True)

        valid_dates = filtered_df["Fecha"].dropna()

        if not valid_dates.empty:
            min_date, max_date = valid_dates.min(), valid_dates.max()
        else:
            min_date, max_date = pd.Timestamp.now() - pd.Timedelta(days=30), pd.Timestamp.now()

        start_date, end_date = st.sidebar.date_input("Rango de fechas:", [min_date.date(), max_date.date()])

        if isinstance(start_date, list) or isinstance(start_date, tuple):
            start_date, end_date = start_date

        filtered_df = filtered_df[(filtered_df["Fecha"] >= pd.to_datetime(start_date)) & (filtered_df["Fecha"] <= pd.to_datetime(end_date))]


    if selected_label != "All":
        filtered_df = filtered_df[filtered_df["etiqueta_predicha"] == selected_label]

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

    st.subheader("📩 Añadir o actualizar correos")

    with st.form("add_email_form"):
        new_text = st.text_area("Escribe el texto del nuevo correo:")
        new_mail = st.text_area("Escribe el correo del remitente:")
        new_label = st.selectbox("Etiqueta (si se conoce):", ["queja", "petición de servicio", "sugerencia de mejora", "desconocida"])
        submitted = st.form_submit_button("Guardar correo")

        if submitted and new_text.strip():
            # Generar embedding
            new_vector = model.encode(new_text).tolist()

            # Crear registro nuevo
            new_row = {
                "Descripción": new_text,
                "etiqueta_predicha": new_label,
                "Email": new_mail,
                "confianza": 1.0 if new_label != "desconocida" else 0.0,
                "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Añadir a DataFrame y guardar CSV
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv("emails_clasificados.csv", index=False)

            # Insertar en Qdrant
            client.upsert(
                collection_name=collection_name,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": new_vector,
                    "payload": {"text": new_text, "label": new_label, "confidence": new_row["confianza"]}
                }]
            )

            st.success("Correo añadido y guardado correctamente.")

    # --- Main table ---
    st.subheader("📊 Todos los emails")
    st.dataframe(filtered_df[["etiqueta_predicha", "confianza", "Descripción"]])


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
