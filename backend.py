import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API keys
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SERPER_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ Please set SERPER_API_KEY and GROQ_API_KEY in your .env file.")

# ✅ Groq models
llm_groq_main = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)   # Raw answer
llm_groq_refine = ChatGroq(model="qwen/qwen3-32b", api_key=GROQ_API_KEY)         # Refined answer

# 🔹 Step 1: Search with Serper.dev
def retrieve_docs(query: str):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    response = requests.post(url, json={"q": query}, headers=headers)
    data = response.json()
    return [{"page_content": r.get("snippet", "")}
            for r in data.get("organic", []) if r.get("snippet")][:5]

# 🔹 Step 2: Get context
def get_context(query: str) -> str:
    docs = retrieve_docs(query)
    return "\n".join(d["page_content"] for d in docs) if docs else "No relevant context found."

# 🔹 Step 3: Generate raw answer with Llama
def generate_with_groq(query: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
You are an AI Doctor assistant.
Use the following context to answer the user's health-related question.
If the question is not health related, still give a helpful and correct answer.

Context:
{context}

Question:
{query}

Raw Answer:
""")
    res = (prompt | llm_groq_main).invoke({"context": context, "query": query})
    return res.content if hasattr(res, "content") else str(res)

# 🔹 Step 4: Refine with Qwen using raw + extra Google search
def refine_with_groq(raw_answer: str, query: str) -> str:
    extra_context = get_context(query)  # fresh google search
    prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable medical AI.
Refine and expand the raw answer using both:
1. The raw answer from another model
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
    return res.content if hasattr(res, "content") else str(res)

# 🔹 Final Answer Flow
def answer_query(query: str):
    context = get_context(query)
    raw = generate_with_groq(query, context)
    refined = refine_with_groq(raw, query)
    return refined  # Only return refined for terminal output

# 🔹 Interactive Chat
if __name__ == "__main__":
    print("🤖 AI Doctor Chatbot (Refined with Google Search + Qwen‑3‑32B)\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            refined = answer_query(q)
            print("\n--- Refined Answer ---")
            print(refined, "\n")
        except Exception as e:
            print("❌ Error:", e, "\n")

