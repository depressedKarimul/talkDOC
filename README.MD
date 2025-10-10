# talkDOC - AI Doctor Chatbot

## Overview
talkDOC is an AI-powered chatbot designed to provide accurate and helpful responses to health-related questions using Groq language models and real-time web search capabilities via the Serper.dev API. It supports text-based and voice-based interactions, with voice input capabilities, language detection/translation for Bangla and English, and image analysis for medical or skin conditions. The chatbot delivers clear, concise, and user-friendly answers through an interactive terminal interface or a Streamlit web interface. Additionally, it processes a disease diagnosis dataset to create a searchable FAISS vector store for enhanced query responses.

## Features
- **Health-Focused Responses**: Provides accurate and fact-checked answers to health-related questions using context from web searches and a FAISS vector store. Only responds to medical-related queries; non-medical questions are declined.
- **Real-Time Web Search**: Utilizes the Serper.dev API to fetch up-to-date information from the top 5 search result snippets.
- **Dataset Processing and Vector Store**: Loads and processes the "sajjadhadi/disease-diagnosis-dataset" Hugging Face dataset, creates text entries combining symptoms and diagnoses, and builds a FAISS vector store using SentenceTransformer embeddings for efficient retrieval.
- **Dual-Model Processing**:
  - Uses `llama-3.3-70b-versatile` for generating raw answers.
  - Refines answers with `qwen/qwen3-32b` for improved accuracy, clarity, and structure.
- **Voice Input**: Supports voice-based queries with speech recognition and language detection/translation (Bangla and English).
- **Image Analysis**: Uploads and analyzes medical or skin images using the `meta-llama/llama-4-scout-17b-16e-instruct` vision model to generate descriptions, which are then classified and processed for medical responses.
- **Medical Question Classification**: Uses `llama-3.1-8b-instant` to classify queries as medical-related before processing.
- **Interactive Interfaces**:
  - **Terminal Interface**: Command-line interaction for text-based queries with refined answers.
  - **Streamlit Web Interface**: User-friendly web app for image upload, voice, and text interactions.
- **Powered by Groq**: Leverages advanced language models for natural language processing.

## Prerequisites
To run talkDOC, you need the following:
- Python 3.8 or higher
- A `.env` file with the following API keys:
  - `SERPER_API_KEY`: Obtain from [Serper.dev](https://serper.dev/)
  - `GROQ_API_KEY`: Obtain from [xAI](https://x.ai/api)
- Internet connection for API requests
- Microphone for voice input (for Streamlit interface)

## Installation
1. **Clone the Repository**:
   ```bash:disable-run
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
   groq
   datasets
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root with the following content:
   ```
   SERPER_API_KEY=your-serper-api-key
   GROQ_API_KEY=your-groq-api-key
   ```

## Usage
talkDOC supports two interfaces: a terminal-based chatbot and a Streamlit web app. Additionally, it includes a script to process the disease diagnosis dataset and create a FAISS vector store for enhanced query responses.

### Terminal Interface
1. **Create FAISS Index** (run once):
   ```bash
   python memory.py
   ```
   This script loads the "sajjadhadi/disease-diagnosis-dataset", prepares symptom-diagnosis text entries, creates embeddings, and saves the FAISS index as `faiss_symptom_disease`.

2. **Run the Chatbot**:
   ```bash
   python backend.py
   ```
   This starts the text-based interactive terminal interface with refined answers.

3. **Interact with talkDOC**:
   - Type your question (e.g., "What are the symptoms of a cold?").
   - The chatbot fetches relevant context from the FAISS vector store, generates a raw answer using `llama-3.3-70b-versatile`, and refines it with `qwen/qwen3-32b` using Google search context for accuracy and clarity.
   - To exit, type `exit` or `quit`.

4. **Example Interaction**:
   ```
   🤖 AI Doctor Chatbot (FAISS → Groq → Refined with Google)

   You: What are the symptoms of a cold?
   --- 🩺 Final Answer ---
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
   - **Image Analysis**: Upload a medical or skin image (JPG, JPEG, PNG) and optionally add a question. Press **Analyze Image** to get a description, classify it as medical, and receive a refined response.
   - **Voice Input**: Record a voice question using the microphone. Press **Send Voice** to transcribe, detect language (Bangla or English), translate to English if Bangla, classify as medical, and display the response (translated back to Bangla if input was Bangla).
   - **Text Input**: Type a question and press **Send Text** for the same processing workflow.
   - Non-medical questions receive a refusal message: "Sorry, I can only answer medical-related questions."

3. **Example Interaction** (Text or Voice):
   - Input: "What are the symptoms of a cold?" (in English or Bangla).
   - Output:
     ```
     🗣️ You asked/said: What are the symptoms of a cold?
     🤖 AI Doctor: Common symptoms of a cold include:
     - Runny or stuffy nose
     - Sore throat
     - Cough
     - Sneezing
     - Mild fatigue
     Summary: A cold causes nasal congestion, sore throat, coughing, sneezing, and mild fatigue, usually resolving in a week or two.
     ```
   - For a non-medical question (e.g., "What is the capital of France?"):
     ```
     🗣️ You asked/said: What is the capital of France?
     🤖 AI Doctor: Sorry, I can only answer medical-related questions.
     ```

### Dataset Processing and Vector Store Creation
1. **Run the Dataset Processing Script**:
   ```bash
   python memory.py
   ```
   This script loads the "sajjadhadi/disease-diagnosis-dataset" from Hugging Face, combines symptoms and diagnoses into text entries, generates embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model, and stores them in a FAISS vector store.

2. **Steps Performed**:
   - Loads the dataset and extracts symptom-diagnosis pairs.
   - Prepares text entries in the format "Symptoms: ... -> Disease: ...".
   - Generates embeddings using the SentenceTransformer model (`all-MiniLM-L6-v2`).
   - Creates a FAISS vector store and saves it locally as `faiss_symptom_disease`.

3. **Output**:
   ```
   📥 Step 1: Loading Symptom-Disease Dataset...
   ✅ Loaded X records from dataset
   🧩 Available columns: ['text', 'diagnosis']
   ✅ Using columns → Symptoms: 'text' | Disease: 'diagnosis'
   🧠 Step 2: Preparing text data...
   ✅ Created Y text entries for embedding
   🧠 Step 3: Loading embedding model → all-MiniLM-L6-v2
   ✅ Embedding model loaded successfully
   📦 Step 4: Creating FAISS index...
   🔄 Embedding Y documents...
   💾 Step 5: FAISS index saved as 'faiss_symptom_disease'
   🎉 All steps completed successfully!
   ```

## Implemented Features in backend.py and frontend.py
The following features have been implemented in the `backend.py` and `frontend.py` scripts to enable the core functionality of talkDOC:

### Backend (backend.py)
- **Environment Setup**: Loads API keys (`SERPER_API_KEY` and `GROQ_API_KEY`) from a `.env` file using `python-dotenv` for secure configuration.
- **Dual LLM Integration**: Utilizes two Groq models:
  - `llama-3.3-70b-versatile` for generating raw answers based on FAISS context.
  - `qwen/qwen3-32b` for refining answers with additional Google search context.
- **FAISS Vector Store**: Loads the pre-built FAISS index (`faiss_symptom_disease`) with SentenceTransformer embeddings (`all-MiniLM-L6-v2`) for retrieving relevant symptom-disease context from the dataset.
- **Google Search Integration**: Uses the Serper.dev API to fetch the top 5 search result snippets for real-time context, enhancing answer accuracy.
- **Prompt Engineering**: Implements `ChatPromptTemplate` for structured prompts to generate raw and refined answers, ensuring clear and professional responses.
- **Answer Refinement**: Combines FAISS context and Google search results to produce fact-checked, concise, and patient-friendly answers with bullet points or steps and a summary.
- **Interactive CLI**: Provides a command-line interface for text-based queries, with error handling and an exit mechanism (`exit` or `quit`).

### Frontend (frontend.py)
- **Streamlit Web Interface**: Creates a user-friendly web app with Streamlit, featuring a centered layout, custom title (`talkDOC - Medical Assistant`), and instructions for image, voice, or text input.
- **Image Upload and Analysis**: Supports image upload (JPG, JPEG, PNG) for medical/skin conditions. Uses `meta-llama/llama-4-scout-17b-16e-instruct` to generate detailed descriptions, classifies them as medical using `llama-3.1-8b-instant`, and passes descriptions to `answer_query` for refined responses.
- **Voice Input**: Supports audio input via Streamlit's `st.audio_input`, converts recordings to WAV format using `pydub` for speech recognition with Google's API via `speechrecognition`.
- **Text Input**: Provides a text input field for typed questions, with the same processing workflow as voice.
- **Language Detection and Translation**: Detects the input language (Bangla or English) using `langdetect` and translates Bangla to English using `deep-translator` with Google Translator. Responses are translated back to Bangla if the input was Bangla.
- **Medical Question Classifier**: Uses the Groq API with the `llama-3.1-8b-instant` model to classify whether a question or image description is health-related. Non-medical inputs receive a polite refusal message.
- **Integration with Backend**: Calls the `answer_query` function from `backend.py` to process health-related queries and retrieve refined AI responses.
- **User Feedback**: Displays transcripts of user input and AI responses, with error handling for invalid inputs (e.g., empty text or no audio) and processing feedback via spinners.

## Code Structure
- **memory.py**: Script for processing the disease diagnosis dataset and creating a FAISS vector store.
  - Loads the dataset using `datasets.load_dataset`.
  - Prepares symptom-diagnosis text entries.
  - Generates embeddings with a custom `SentenceTransformerEmbeddings` class.
  - Creates and saves a FAISS vector store using `langchain_community.vectorstores.FAISS`.
- **backend.py**: Main script for the terminal-based chatbot, handling LLM-related logic.
  - `retrieve_docs_google`: Fetches top 5 search result snippets using Serper.dev API.
  - `get_google_context`: Combines snippets into a context string.
  - `get_vector_context`: Retrieves context from FAISS vector store.
  - `generate_with_groq`: Generates raw answers using `llama-3.3-70b-versatile`.
  - `refine_with_groq`: Refines answers using `qwen/qwen3-32b` with additional Google context.
  - `answer_query`: Combines context retrieval, raw answer generation, and refinement.
  - Interactive loop for terminal-based chat.
- **frontend.py**: Script for the Streamlit web interface.
  - Handles image upload and analysis with Groq vision model.
  - Handles voice input, speech-to-text, language detection, translation, and medical classification.
  - Handles text input with the same workflow.
  - Integrates with `answer_query` from `backend.py` for AI responses.
- **.env**: Stores API keys for Serper.dev and Groq.
- **requirements.txt**: Lists required Python packages.

## Dependencies
- `requests`: For making HTTP requests to the Serper.dev API.
- `python-dotenv`: For loading environment variables from a `.env` file.
- `langchain-groq`: For integrating with Groq language models (`llama-3.3-70b-versatile` and `qwen/qwen3-32b`).
- `langchain-core`: For building prompt and chain logic.
- `langchain-community`: For FAISS vector store functionality.
- `sentence-transformers`: For generating embeddings using the `all-MiniLM-L6-v2` model.
- `faiss-cpu`: For creating and managing the FAISS vector store.
- `streamlit`: For the web-based user interface.
- `speechrecognition`: For converting voice input to text.
- `pydub`: For audio file processing.
- `langdetect`: For detecting the language of user input.
- `deep-translator`: For translating between Bangla and English.
- `groq`: For direct interaction with the Groq API for classification and vision analysis.
- `datasets`: For loading the Hugging Face disease diagnosis dataset.

## Limitations
- **Health Advice**: talkDOC is not a substitute for professional medical advice. Always consult a licensed healthcare provider for medical concerns.
- **API Dependency**: Requires valid API keys and an internet connection.
- **Search Results**: Limited to the top 5 snippets from Serper.dev, which may not always cover all relevant information.
- **Vector Store**: The FAISS vector store is limited to the content of the disease diagnosis dataset and the quality of embeddings.
- **Voice Input**: Requires a working microphone and may be affected by background noise or accents. Speech recognition uses Google's API, which may have limitations for Bangla.
- **Image Analysis**: Supports only JPG, JPEG, PNG formats; focused on medical/skin images; relies on vision model accuracy.
- **Language Support**: Currently supports Bangla and English; detection and translation may not be perfect for mixed or informal text. Non-medical refusal messages are in English only.
- **Question Scope**: Only responds to health-related questions due to the medical question classifier; non-medical queries are declined.

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
```