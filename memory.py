import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # progress bar

# 1. Load environment variables
load_dotenv()

print("📂 Step 1: Loading PDF...")
loader = PyPDFLoader("data/disease_symptoms.pdf")
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages from PDF")

# 2. Split into chunks
print("✂️ Step 2: Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks")

# 3. SentenceTransformer wrapper
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🧠 Step 3: Loading embedding model → {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Model loaded successfully")

    def embed_documents(self, texts):
        print(f"🔄 Embedding {len(texts)} documents...")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# 4. Initialize embedding object
embedding = SentenceTransformerEmbeddings()

# 5. Store in FAISS
print("📦 Step 4: Creating FAISS index...")
vectorstore = FAISS.from_documents(tqdm(chunks, desc="🔍 Embedding Chunks"), embedding=embedding)
print("✅ FAISS index created")

# 6. Save index locally
vectorstore.save_local("faiss_index_sentence")
print("💾 Step 5: FAISS index saved as 'faiss_index_sentence'")
print("🎉 All steps completed successfully!")
