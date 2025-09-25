import os
import re
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# -----------------------------
# Load API keys
# -----------------------------
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SERPER_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ Please set SERPER_API_KEY and GROQ_API_KEY in your .env file.")

# -----------------------------
# Groq LLM models
# -----------------------------
llm_groq_main = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)   # Raw answer
llm_groq_refine = ChatGroq(model="qwen/qwen3-32b", api_key=GROQ_API_KEY)         # Refined answer

# -----------------------------
# Custom Embeddings class (same as memory.py)
# -----------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🧠 Loading embedding model → {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded")

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# -----------------------------
# Load FAISS vectorstore with embeddings
# -----------------------------
print("📂 Loading FAISS index from local...")
embedding = SentenceTransformerEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index_sentence",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("✅ FAISS retriever ready")

# -----------------------------
# Google Search for refining answers
# -----------------------------
def retrieve_docs_google(query: str):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    response = requests.post(url, json={"q": query}, headers=headers)
    data = response.json()
    return [{"page_content": r.get("snippet", "")}
            for r in data.get("organic", []) if r.get("snippet")][:5]

def get_google_context(query: str) -> str:
    docs = retrieve_docs_google(query)
    return "\n".join(d["page_content"] for d in docs) if docs else "No relevant context found."

# -----------------------------
# Get context from FAISS (raw answer)
# -----------------------------
def get_vector_context(query: str) -> str:
    # Using invoke() to avoid deprecation warning
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs) if docs else "No relevant context in database."

# -----------------------------
# Generate raw answer using FAISS context
# -----------------------------
def generate_with_groq(query: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
You are an AI Doctor assistant.
Use the following context from the knowledge base to answer the user's health-related question.
If the question is not health related, still give a helpful and correct answer.

Context (from FAISS):
{context}

Question:
{query}

Raw Answer:
""")
    res = (prompt | llm_groq_main).invoke({"context": context, "query": query})
    return res.content if hasattr(res, "content") else str(res)

# -----------------------------
# Refine raw answer using Google search
# -----------------------------
def refine_with_groq(raw_answer: str, query: str) -> str:
    extra_context = get_google_context(query)
    prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable medical AI.
Refine and expand the raw answer using both:
1. The raw answer from another model (based on FAISS knowledge base)
2. Additional verified context from Google search

Requirements:
- Make answer accurate and fact-checked
- Clear and concise
- Easy to understand for a patient
- Structured: use steps or bullet points if needed
- Include a short summary at the end
- Professional but friendly tone

Raw Answer:
{raw_answer}

Extra Google Context:
{extra_context}

Final Refined Answer:
""")
    res = (prompt | llm_groq_refine).invoke({"raw_answer": raw_answer, "extra_context": extra_context})
    refined = res.content if hasattr(res, "content") else str(res)
    refined = re.sub(r"<think>.*?</think>", "", refined, flags=re.DOTALL).strip()
    return refined

# -----------------------------
# Final answer flow
# -----------------------------
def answer_query(query: str) -> str:
    context = get_vector_context(query)    # FAISS only
    raw = generate_with_groq(query, context)
    refined = refine_with_groq(raw, query) # Refined with Google
    return refined

# -----------------------------
# Interactive CLI
# -----------------------------
if __name__ == "__main__":
    print("🤖 AI Doctor Chatbot (FAISS → Groq → Refined with Google)\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            refined = answer_query(q)
            print("\n--- Final Answer ---")
            print(refined, "\n")
        except Exception as e:
            print("❌ Error:", e, "\n")
