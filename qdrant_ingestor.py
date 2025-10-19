from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os
import uuid
import pandas as pd

load_dotenv()  # loads the .env file
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=url, api_key=api_key)

collection_name = "emailclassification_test"

# df = pd.read_csv("emails_clasificados.csv")

# # print(df.columns)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# embeddings = model.encode(df["Descripción"].tolist(), show_progress_bar=True)

# client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE) # WHY COSINE DISTANCE AND NOT DOT PRODUCT
# )

# points = [
#     PointStruct(
#         id=str(uuid.uuid4()),
#         vector=embedding.tolist(),
#         payload={
#             "text": row["Descripción"],
#             "label": row["etiqueta_predicha"],
#             "confidence": float(row["confianza"])
#         }
#     )
#     for embedding, (_, row) in zip(embeddings, df.iterrows())
# ]

# operation_info = client.upsert(collection_name=collection_name, points=points)

# print(operation_info)

# print(f" Inserted {len(points)} emails into Qdrant collection '{collection_name}'")


# --- RUN A QUERY ---

query = "hay algun correo de quejas?"

query_vector = model.encode(query).tolist()

results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3,
    with_payload=True
)

print(results)
