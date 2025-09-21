import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# === Step 1: Load PDF ===
PDF_FILE = "data/disease_symptoms.pdf"
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()
print(f"[INFO] Loaded {len(documents)} pages from {PDF_FILE}")

# === Step 2: Split text into chunks ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", ",", " "]
)
chunks = splitter.split_documents(documents)
print(f"[INFO] Created {len(chunks)} text chunks.")

# === Step 3: Use HuggingFaceEmbeddings WITHOUT sentence-transformers ===
EMBEDDING_MODEL_NAME = "distilbert-base-uncased"   # lightweight, TF-free
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},        # or "cuda" if you have GPU
    encode_kwargs={"normalize_embeddings": True}
)
print(f"[INFO] Using embedding model: {EMBEDDING_MODEL_NAME}")

# === Step 4: Build & Save FAISS Vector Store ===
DB_FAISS_PATH = "vectorstore/db_faiss_pdf"
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"[SUCCESS] FAISS vector store saved at: {DB_FAISS_PATH}")
