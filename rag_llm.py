import os
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ======================
# CONFIGURACIÓN INICIAL
# ======================

load_dotenv()
client = OpenAI()
history = []

# Modelo de embeddings (mismo usado en la ingesta)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Conexión con Qdrant (API cloud o local)
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "emailclassification_test"  # Asegúrate de que coincida con el nombre usado antes


# ======================
# FUNCIÓN DE CONSULTA RAG
# ======================

def query_qdrant(user_query, top_k=3):
    """Busca los textos más similares en Qdrant y devuelve su payload."""
    query_vector = embedder.encode(user_query).tolist()

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    # Extraemos los textos de los resultados
    context_chunks = [r.payload.get("text", "") for r in results]
    return context_chunks


# ======================
# FUNCIÓN DE CONVERSACIÓN CON EL LLM
# ======================

def send_user_message(user_msg, retrieved_context=None):
    """
    Envía un mensaje al LLM (Ricky Gervais versión consultor técnico)
    con contexto opcional del RAG.
    """

    # Si hay contexto, lo formateamos tipo "Mail 1", "Mail 2", etc.
    if retrieved_context:
        if isinstance(retrieved_context, list):
            formatted_context = "\n\n".join(
                [f"Mail {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_context)]
            )
        else:
            formatted_context = f"Mail 1:\n{retrieved_context}"

        user_msg = (
            f"Contexto recuperado del sistema RAG:\n{formatted_context}\n\n"
            f"Pregunta del usuario:\n{user_msg}"
        )

    history.append({"role": "user", "content": user_msg})

    input_for_api = [{"role": msg["role"], "content": msg["content"]} for msg in history]

    stream = client.responses.create(
        model="gpt-4.1-nano",
        instructions=(
            "Eres Ricky Gervais atrapado en el cuerpo de un asistente de IA en el correo de una persona. "
            "Tienes un humor ácido, sarcástico y un punto cruel."
            "Recibes contexto desde un sistema de Recuperación Aumentada (RAG), que contiene información "
            "útil para responder preguntas del usuario. Este contexto se presenta como 'Mail 1', 'Mail 2', etc. "
            "Úsalo si tiene sentido, ridiculízalo si es irrelevante, "
            "pero siempre responde con precisión técnica y claridad. "
            "Tu tono debe ser directo, irónico y con la confianza de alguien que ha visto demasiada estupidez humana."
            "Cuando el usuario te pregunte algo de correos o mails, se claro en tu respuesta."
        ),
        input=input_for_api,
        stream=True,
    )

    assistant_msg = ""
    msg_tokens = 0
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end='', flush=True)
            assistant_msg += event.delta
        if event.type == "response.completed":
            msg_tokens = event.response.usage.total_tokens
    print()

    history.append({
        "role": "assistant",
        "content": assistant_msg,
        "msg_tokens": msg_tokens,
    })


# ======================
# EJECUCIÓN PRINCIPAL
# ======================

if __name__ == "__main__":
    print("🤖 Bienvenido al asistente RAG (versión Ricky Gervais)\n")

    while True:
        user_input = input("\nTú: ")
        print('\n')
        if user_input.lower() in ["exit", "quit", "salir"]:
            print("Vale, adiós. Ve a tocar hierba o algo.")
            break

        # Buscar contexto en Qdrant
        context = query_qdrant(user_input)

        # Enviar mensaje al modelo con contexto
        send_user_message(user_input, retrieved_context=context)
        print('\n')
        print(context)