# talkDOC - AI Doctor Chatbot

## Overview
talkDOC is an AI-powered chatbot designed to provide accurate and helpful responses to health-related questions using the Groq language model and real-time web search capabilities via the Serper.dev API. It supports both text-based and voice-based interactions, with voice input/output capabilities and language translation for Bangla and English. The chatbot delivers clear, concise, and user-friendly answers through an interactive terminal interface or a Streamlit web interface.

## Features
- **Health-Focused Responses**: Provides accurate and fact-checked answers to health-related questions using context from web searches.
- **General Query Support**: Answers non-health-related questions with relevant and verified information.
- **Real-Time Web Search**: Utilizes the Serper.dev API to fetch up-to-date information from the top 5 search result snippets.
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
   streamlit
   speechrecognition
   pydub
   langdetect
   deep-translator
   gtts
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root with the following content:
   ```
   SERPER_API_KEY=your-serper-api-key
   GROQ_API_KEY=your-groq-api-key
   ```

## Usage
talkDOC supports two interfaces: a terminal-based chatbot and a Streamlit web app.

### Terminal Interface
1. **Run the Chatbot**:
   ```bash
   python backend.py
   ```
   This starts the text-based interactive terminal interface with refined answers.

2. **Interact with talkDOC**:
   - Type your question (e.g., "What are the symptoms of a cold?").
   - The chatbot fetches relevant context from the top 5 web search snippets, generates a raw answer using `llama-3.3-70b-versatile`, and refines it with `qwen/qwen3-32b` for accuracy and clarity.
   - To exit, type `exit` or `quit`.

3. **Example Interaction**:
   ```
   🤖 AI Doctor Chatbot (Refined with Google Search + Qwen-3-32B)

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
   - Record a voice question using the microphone input.
   - Press the **Send** button to process the query.
   - The app detects the language (Bangla or English), translates to English if necessary, queries the AI, and provides a refined response in text and audio.
   - View the transcript of your question and the AI's response on the screen.

3. **Example Interaction**:
   - Record: "What are the symptoms of a cold?" (in English or Bangla).
   - After pressing **Send**, the app displays:
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

## Code Structure
- **backend.py**: Main script for the terminal-based chatbot, handling LLM-related logic.
  - `retrieve_docs`: Fetches top 5 search result snippets using Serper.dev API.
  - `get_context`: Combines snippets into a context string.
  - `generate_with_groq`: Generates raw answers using `llama-3.3-70b-versatile`.
  - `refine_with_groq`: Refines answers using `qwen/qwen3-32b` with additional context.
  - `answer_query`: Combines context retrieval, raw answer generation, and refinement.
  - Interactive loop for terminal-based chat.
- **frontend.py**: Script for the Streamlit web interface.
  - Handles voice input, speech-to-text, language detection, translation, and text-to-speech.
  - Integrates with `answer_query` from `backend.py` for AI responses.
- **.env**: Stores API keys for Serper.dev and Groq.
- **requirements.txt**: Lists required Python packages.

## Dependencies
- `requests`: For making HTTP requests to the Serper.dev API.
- `python-dotenv`: For loading environment variables from a `.env` file.
- `langchain-groq`: For integrating with Groq language models (`llama-3.3-70b-versatile` and `qwen/qwen3-32b`).
- `langchain-core`: For building prompt and chain logic.
- `streamlit`: For the web-based user interface.
- `speechrecognition`: For converting voice input to text.
- `pydub`: For audio file processing.
- `langdetect`: For detecting the language of user input.
- `deep-translator`: For translating between Bangla and English.
- `gtts`: For generating text-to-speech audio responses.

## Limitations
- **Health Advice**: talkDOC is not a substitute for professional medical advice. Always consult a licensed healthcare provider for medical concerns.
- **API Dependency**: Requires valid API keys and an internet connection.
- **Search Results**: Limited to the top 5 snippets from Serper.dev, which may not always cover all relevant information.
- **Voice Input**: Requires a working microphone and may be affected by background noise or accents.
- **Language Support**: Currently supports Bangla and English; other languages may require additional configuration.

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