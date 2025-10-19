from transformers import pipeline
import pandas as pd

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

df = pd.read_csv("dataset_prueba.csv", sep=";")
labels = ["queja", "petición de servicio", "sugerencia de mejora"]

preds, confs = [], []

for text in df["Descripción"].fillna(""):
    result = classifier(text, candidate_labels=labels, hypothesis_template="Este texto es una {}.")
    preds.append(result["labels"][0])
    confs.append(result["scores"][0])

df["etiqueta_predicha"] = preds
df["confianza"] = confs
df.to_csv("emails_clasificados.csv", index=False)
print("✅ Done: saved emails_clasificados.csv")
