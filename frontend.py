import streamlit as st
from backend import answer_query  # Import the answer_query function from your backend

# Streamlit page configuration
st.set_page_config(page_title="talkDOC - AI Doctor Chatbot", page_icon="🩺", layout="centered")

# Custom CSS for a modern, health-themed design with professional fonts
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    .user-message {
        color: #000000;
        background-color: #d1e7dd;
        border-left: 5px solid #4CAF50;
    }
    .ai-message {
        background-color: #b3d4fc; /* Darker blue for better contrast */
        border-left: 5px solid #2196F3;
        color: #000000; /* Black text for readability */
        font-family: 'Helvetica', 'Arial', sans-serif; /* Professional font */
    }
    .ai-message p {
        margin: 10px 0;
        line-height: 1.5;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 20px;
        font-size: 14px;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and header
st.title("🩺 talkDOC - Your AI Doctor Assistant")
st.markdown("Ask health-related questions, and I'll provide clear, helpful answers based on web information.")

# Input form for user query
with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input("Your Question:", placeholder="E.g., What are symptoms of a cold?")
    submit_button = st.form_submit_button("Ask")

# Handle query submission
if submit_button and user_query:
    try:
        # Get AI response from backend
        ai_response = answer_query(user_query)
        # Append to chat history
        st.session_state.chat_history.append({"user": user_query, "ai": ai_response})
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Display chat history
for chat in st.session_state.chat_history:
    # User message
    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
    # AI response with text blocks
    st.markdown(
        f'<div class="chat-message ai-message"><strong>AI Doctor:</strong>'
        f'<p>{chat["ai"].replace("\n", "</p><p>")}</p></div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown('<div class="footer">Powered by talkDOC | Not a substitute for professional medical advice</div>', unsafe_allow_html=True)