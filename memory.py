import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()

print("📥 Step 1: Loading Symptom-Disease Dataset...")
dataset = load_dataset("sajjadhadi/disease-diagnosis-dataset")
data = dataset["train"]
print(f"✅ Loaded {len(data)} records from dataset")

# -----------------------------
# 2️⃣ Column names
# -----------------------------
print(f"🧩 Available columns: {data.column_names}")
symptom_col = "text"
disease_col = "diagnosis"  
print(f"✅ Using columns → Symptoms: '{symptom_col}' | Disease: '{disease_col}'")

# -----------------------------
# 3️⃣ Prepare text data
# -----------------------------
print("🧠 Step 2: Preparing text data...")
texts = [
    f"Symptoms: {item[symptom_col]} -> Disease: {item[disease_col]}"
    for item in data
]
print(f"✅ Created {len(texts)} text entries for embedding")

# -----------------------------
# 4️⃣ SentenceTransformer embedding wrapper
# -----------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🧠 Step 3: Loading embedding model → {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully")

    def embed_documents(self, texts):
        print(f"🔄 Embedding {len(texts)} documents...")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# -----------------------------
# 5️⃣ Initialize embedding
# -----------------------------
embedding = SentenceTransformerEmbeddings()

# -----------------------------
# 6️⃣ Create and save FAISS index
# -----------------------------
print("📦 Step 4: Creating FAISS index...")
vectorstore = FAISS.from_texts(texts, embedding=embedding)
vectorstore.save_local("faiss_symptom_disease")

print("💾 Step 5: FAISS index saved as 'faiss_symptom_disease'")
print("🎉 All steps completed successfully!")
