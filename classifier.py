from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import random

# Fix seed across all libraries
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

classifier = pipeline(
    "zero-shot-classification",
    # model="facebook/bart-large-mnli"
    model = "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli",
    device=0 if torch.cuda.is_available() else -1
)

classifier.model.eval()

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
print("Done: saved emails_clasificados.csv")
