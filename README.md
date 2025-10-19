# 📬 Email Classification & RAG Assistant (Fenie Challenge)

A multilingual email classification and retrieval-augmented generation (RAG) system that classifies user emails, stores them in a Qdrant vector database, and lets you explore, search, and chat with a sarcastic LLM-powered assistant (“Ricky Gervais”) about the data.

## 🧠 Overview

This project combines **zero-shot classification**, **semantic search**, and **retrieval-augmented generation (RAG)** to create an intelligent dashboard for analyzing and interacting with customer emails.

It includes:
- **Streamlit UI** for interactive dashboards and chat.
- **Email classification** using Hugging Face transformers.
- **Qdrant vector storage** for semantic search.
- **RAG Assistant** using an OpenAI model with a sarcastic personality.

---

## 🧩 Project Structure

```
├── app.py                  # Streamlit app for dashboard + RAG chat
├── classifier.py           # Zero-shot classifier for labeling emails
├── qdrant_ingestor.py      # Embedding and ingestion script for Qdrant
├── rag_llm.py              # RAG logic + OpenAI integration
├── dataset_prueba.csv      # Input dataset with raw email texts
└── emails_clasificados.csv # (Generated) Classified emails with labels
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fenie-challenge-genai.git
cd fenie-challenge-genai
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

*(If there’s no requirements file yet, install manually:)*
```bash
pip install streamlit qdrant-client sentence-transformers openai python-dotenv transformers torch pandas numpy
```

### 4. Add your environment variables
Create a `.env` file in the project root:
```
QDRANT_URL=<your-qdrant-endpoint>
QDRANT_API_KEY=<your-qdrant-api-key>
OPENAI_API_KEY=<your-openai-key>
```

---

## 🧮 Usage

### Step 1: Classify Emails
Run the classifier to generate predicted labels and confidence scores:
```bash
python classifier.py
```
This creates `emails_clasificados.csv`.

### Step 2: Ingest Data into Qdrant
Run the ingestion script to store email embeddings:
```bash
python qdrant_ingestor.py
```

### Step 3: Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🖥️ Features

### 📊 **Dashboard**
- View and filter classified emails by **label**, **date**, or **semantic similarity**.
- Add or update emails manually.
- Search through emails using multilingual embeddings.

### 💬 **RAG Assistant**
- Ask questions about the dataset.
- Uses retrieved emails as context for responses.
- Responds in sarcastic, Ricky Gervais–style commentary while staying technically accurate.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Interface | Streamlit |
| Embeddings | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Classification | Hugging Face Zero-Shot Pipeline (`MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli`) |
| Vector Database | Qdrant |
| Language Model | OpenAI GPT-4.1 Nano |
| Environment | Python 3.9+ |

