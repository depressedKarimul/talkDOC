# talkDOC - AI Doctor Chatbot

## Overview
talkDOC is an AI-powered chatbot designed to provide accurate and helpful responses to health-related questions using the Groq language model and real-time web search capabilities via the Serper.dev API. It supports both text-based and voice-based interactions, with voice input/output capabilities and language translation for Bangla and English. The chatbot delivers clear, concise, and user-friendly answers through an interactive terminal interface or a Streamlit web interface. Additionally, it processes health-related PDFs to create a searchable FAISS vector store for enhanced query responses.

## Features
- **Health-Focused Responses**: Provides accurate and fact-checked answers to health-related questions using context from web searches and a FAISS vector store.
- **General Query Support**: Answers non-health-related questions with relevant and verified information.
- **Real-Time Web Search**: Utilizes the Serper.dev API to fetch up-to-date information from the top 5 search result snippets.
- **PDF Processing and Vector Store**: Loads and processes health-related PDFs (e.g., `disease_symptoms.pdf`), splits them into chunks, and creates a FAISS vector store using SentenceTransformer embeddings for efficient retrieval.
- **Dual-Model Processing**:
  - Uses `llama-3.3-70b-versatile` for generating raw answers.
  - Refines answers with `qwen/qwen3-32b` for improved accuracy, clarity, and structure.
- **Voice Input/Output**: Supports voice-based queries and responses with speech recognition, text-to-speech, and language detection/translation (Bangla and English).
- **Interactive Interfaces**:
  - **Terminal Interface**: Command-line interaction for text-based queries with refined answers.
  - **Streamlit Web Interface**: User-friendly web app for voice and text interactions.
- **Powered by Groq**: Leverages advanced language models for natural language processing.

## Prerequisites
To run talkDOC, you need the following:
- Python 3.8 or higher
- A `.env` file with the following API keys:
  - `SERPER_API_KEY`: Obtain from [Serper.dev](https://serper.dev/)
  - `GROQ_API_KEY`: Obtain from [xAI](https://x.ai/api)
- Internet connection for API requests
- Microphone for voice input (for Streamlit interface)
- Health-related PDF files (e.g., `disease_symptoms.pdf`) in the `data/` directory for vector store creation

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/depressedKarimul/talkDOC.git
   cd talkDOC
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure the following packages are included in `requirements.txt`:
   ```
   requests
   python-dotenv
   langchain-groq
   langchain-core
   langchain-community
   sentence-transformers
   faiss-cpu
   streamlit
   speechrecognition
   pydub
   langdetect
   deep-translator
   gtts
   groq
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root with the following content:
   ```
   SERPER_API_KEY=your-serper-api-key
   GROQ_API_KEY=your-groq-api-key
   ```

5. **Prepare PDF Data**:
   Place health-related PDF files (e.g., `disease_symptoms.pdf`) in the `data/` directory for processing.

## Usage
talkDOC supports two interfaces: a terminal-based chatbot and a Streamlit web app. Additionally, it includes a script to process PDFs and create a FAISS vector store for enhanced query responses.

### Terminal Interface
1. **Run the Chatbot**:
   ```bash
   python backend.py
   ```
   This starts the text-based interactive terminal interface with refined answers.

2. **Interact with talkDOC**:
   - Type your question (e.g., "What are the symptoms of a cold?").
   - The chatbot fetches relevant context from the top 5 web search snippets and the FAISS vector store, generates a raw answer using `llama-3.3-70b-versatile`, and refines it with `qwen/qwen3-32b` for accuracy and clarity.
   - To exit, type `exit` or `quit`.

3. **Example Interaction**:
   ```
   🤖 AI Doctor Chatbot (Refined with Google Search + Qwen-3-32B + FAISS)

   You: What are the symptoms of a cold?
   --- Refined Answer ---
   Common symptoms of a cold include:
   - Runny or stuffy nose
   - Sore throat
   - Cough
   - Sneezing
   - Mild fatigue
   Summary: A cold causes nasal congestion, sore throat, coughing, sneezing, and mild fatigue, usually resolving in a week or two.

   You: exit
   👋 Goodbye!
   ```

### Streamlit Web Interface
1. **Run the Web App**:
   ```bash
   streamlit run frontend.py
   ```
   This launches the web interface in your default browser.

2. **Interact with talkDOC**:
   - Record a voice question using the microphone input or type a question in the text input field.
   - Press the **Send Voice** or **Send Text** button to process the query.
   - The app detects the language (Bangla or English), translates to English if necessary, checks if the question is health-related, and queries the AI with context from the FAISS vector store and web search, providing a refined response in text and audio.
   - Non-medical questions receive a message indicating that only health-related queries are supported.
   - View the transcript of your question and the AI's response on the screen.

3. **Example Interaction**:
   - Record or type: "What are the symptoms of a cold?" (in English or Bangla).
   - After pressing **Send Voice** or **Send Text**, the app displays:
     ```
     🗣️ You said: What are the symptoms of a cold?
     🤖 AI Doctor: Common symptoms of a cold include:
     - Runny or stuffy nose
     - Sore throat
     - Cough
     - Sneezing
     - Mild fatigue
     Summary: A cold causes nasal congestion, sore throat, coughing, sneezing, and mild fatigue, usually resolving in a week or two.
     ```
   - An audio response is played in the detected language.
   - For a non-medical question (e.g., "What is the capital of France?"):
     ```
     🗣️ You said: What is the capital of France?
     🤖 AI Doctor: I’m sorry, I can only provide medical and health-related suggestions. I won’t be able to answer questions outside this topic.
     ```

### PDF Processing and Vector Store Creation
1. **Run the PDF Processing Script**:
   ```bash
   python memory.py
   ```
   This script processes a PDF file (e.g., `disease_symptoms.pdf`), splits it into chunks, creates embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model, and stores them in a FAISS vector store.

2. **Steps Performed**:
   - Loads the PDF from the `data/` directory.
   - Splits the PDF content into chunks (1000 characters with 100-character overlap).
   - Generates embeddings using the SentenceTransformer model (`all-MiniLM-L6-v2`).
   - Creates a FAISS vector store and saves it locally as `faiss_index_sentence`.

3. **Output**:
   ```
   📂 Step 1: Loading PDF...
   ✅ Loaded X pages from PDF
   ✂️ Step 2: Splitting text into chunks...
   ✅ Created Y chunks
   🧠 Step 3: Loading embedding model → all-MiniLM-L6-v2
   ✅ Model loaded successfully
   📦 Step 4: Creating FAISS index...
   🔍 Embedding Chunks: 100%|█████████████████████| Y/Y [00:XX<00:01, Z chunks/s]
   ✅ FAISS index created
   💾 Step 5: FAISS index saved as 'faiss_index_sentence'
   🎉 All steps completed successfully!
   ```

## Implemented Features in backend.py and frontend.py
The following features have been implemented in the `backend.py` and `frontend.py` scripts to enable the core functionality of talkDOC:

### Backend (backend.py)
- **Environment Setup**: Loads API keys (`SERPER_API_KEY` and `GROQ_API_KEY`) from a `.env` file using `python-dotenv` for secure configuration.
- **Dual LLM Integration**: Utilizes two Groq models:
  - `llama-3.3-70b-versatile` for generating raw answers based on FAISS context.
  - `qwen/qwen3-32b` for refining answers with additional Google search context.
- **FAISS Vector Store**: Loads a pre-built FAISS index (`faiss_index_sentence`) with SentenceTransformer embeddings (`all-MiniLM-L6-v2`) for retrieving relevant health-related context from processed PDFs.
- **Google Search Integration**: Uses the Serper.dev API to fetch the top 5 search result snippets for real-time context, enhancing answer accuracy.
- **Prompt Engineering**: Implements `ChatPromptTemplate` for structured prompts to generate raw and refined answers, ensuring clear and professional responses.
- **Answer Refinement**: Combines FAISS context and Google search results to produce fact-checked, concise, and patient-friendly answers with bullet points or steps and a summary.
- **Interactive CLI**: Provides a command-line interface for text-based queries, with error handling and an exit mechanism (`exit` or `quit`).

### Frontend (frontend.py)
- **Streamlit Web Interface**: Creates a user-friendly web app with Streamlit, featuring a centered layout, custom title (`talkDOC - Voice Doctor Assistant`), and a footer disclaimer.
- **Voice Input**: Supports audio input via Streamlit's `st.audio_input`, storing recordings temporarily in session state and converting them to WAV format using `pydub` for speech recognition.
- **Speech-to-Text**: Uses `speechrecognition` with Google's speech recognition API to convert audio input to text.
- **Language Detection and Translation**: Detects the input language (Bangla or English) using `langdetect` and translates Bangla to English (and vice versa for responses) using `deep-translator` with Google Translator.
- **Medical Question Classifier**: Uses the Groq API with the `llama-3.1-8b-instant` model to classify whether a question is health-related (e.g., about diseases, symptoms, treatments). Non-medical questions receive a polite refusal message in the detected language (e.g., "I’m sorry, I can only provide medical and health-related suggestions" in English or "Sorry, ami bairer question er answer dite parbo na" in Bangla).
- **Text-to-Speech**: Generates audio responses using `gTTS` in the detected language (Bangla or English), saved as temporary MP3 files and played via Streamlit's `st.audio`.
- **Text Input**: Provides an alternative text input field for users to type questions, with the same language detection, translation, and classification workflow.
- **Integration with Backend**: Calls the `answer_query` function from `backend.py` to process health-related queries and retrieve refined AI responses.
- **User Feedback**: Displays transcripts of user input and AI responses, with error handling for invalid inputs (e.g., empty text or no audio) and processing feedback via spinners.

## Code Structure
- **backend.py**: Main script for the terminal-based chatbot, handling LLM-related logic.
  - `retrieve_docs_google`: Fetches top 5 search result snippets using Serper.dev API.
  - `get_google_context`: Combines snippets into a context string.
  - `get_vector_context`: Retrieves context from FAISS vector store.
  - `generate_with_groq`: Generates raw answers using `llama-3.3-70b-versatile`.
  - `refine_with_groq`: Refines answers using `qwen/qwen3-32b` with additional Google context.
  - `answer_query`: Combines context retrieval, raw answer generation, and refinement.
  - Interactive loop for terminal-based chat.
- **frontend.py**: Script for the Streamlit web interface.
  - Handles voice input, speech-to-text, language detection, translation, medical question classification, and text-to-speech.
  - Integrates with `answer_query` from `backend.py` for AI responses.
- **pdf_processor.py**: Script for processing PDFs and creating a FAISS vector store.
  - Loads and splits PDF content using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
  - Generates embeddings with a custom `SentenceTransformerEmbeddings` class.
  - Creates and saves a FAISS vector store using `langchain_community.vectorstores.FAISS`.
- **.env**: Stores API keys for Serper.dev and Groq.
- **requirements.txt**: Lists required Python packages.

## Dependencies
- `requests`: For making HTTP requests to the Serper.dev API.
- `python-dotenv`: For loading environment variables from a `.env` file.
- `langchain-groq`: For integrating with Groq language models (`llama-3.3-70b-versatile` and `qwen/qwen3-32b`).
- `langchain-core`: For building prompt and chain logic.
- `langchain-community`: For PDF loading and FAISS vector store functionality.
- `sentence-transformers`: For generating embeddings using the `all-MiniLM-L6-v2` model.
- `faiss-cpu`: For creating and managing the FAISS vector store.
- `streamlit`: For the web-based user interface.
- `speechrecognition`: For converting voice input to text.
- `pydub`: For audio file processing.
- `langdetect`: For detecting the language of user input.
- `deep-translator`: For translating between Bangla and English.
- `gtts`: For generating text-to-speech audio responses.
- `groq`: For direct interaction with the Groq API for medical question classification.

## Limitations
- **Health Advice**: talkDOC is not a substitute for professional medical advice. Always consult a licensed healthcare provider for medical concerns.
- **API Dependency**: Requires valid API keys and an internet connection.
- **Search Results**: Limited to the top 5 snippets from Serper.dev, which may not always cover all relevant information.
- **Vector Store**: The FAISS vector store is limited to the content of provided PDFs and the quality of embeddings.
- **Voice Input**: Requires a working microphone and may be affected by background noise or accents.
- **Language Support**: Currently supports Bangla and English; other languages may require additional configuration.
- **Question Scope**: Only responds to health-related questions due to the medical question classifier; non-medical queries are politely declined.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, please open an issue on the repository or contact the project maintainers.