import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API keys from .env
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SERPER_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ Please set SERPER_API_KEY and GROQ_API_KEY in your .env file.")

# ✅ Groq LLM (latest supported model)
llm_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# 🔹 Step 1: Search with Serper.dev
def retrieve_docs(query: str):
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    # Extract top 3 snippets
    documents = [
        {"page_content": r.get("snippet", "")}
        for r in data.get("organic", [])
        if r.get("snippet")
    ][:3]

    return documents


# 🔹 Step 2: Get context from docs
def get_context(query: str) -> str:
    docs = retrieve_docs(query)
    if not docs:
        return "No relevant context found."
    context_text = "\n".join([d["page_content"] for d in docs])
    return context_text


# 🔹 Step 3: Answer query with Groq LLM
def answer_query(query: str) -> str:
    context = get_context(query)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI Doctor assistant. 
        Use the following context to answer the user's health-related question. 
        If the question is not health related, still give a helpful answer.

        Context:
        {context}

        Question:
        {query}

        Answer clearly, simply, and in a helpful tone:
        """
    )

    chain = prompt | llm_model
    answer = chain.invoke({"context": context, "query": query})
    return answer.content


# 🔹 Interactive Terminal Chat
if __name__ == "__main__":
    print("🤖 AI Doctor Chatbot (type 'exit' to quit)\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            ans = answer_query(q)
            print("AI Doctor:", ans, "\n")
        except Exception as e:
            print("❌ Error:", e, "\n")
