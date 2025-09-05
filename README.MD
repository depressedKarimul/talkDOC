# talkDOC - AI Doctor Chatbot

## Overview
talkDOC is an AI-powered chatbot designed to provide helpful responses to health-related questions using the Groq language model and real-time web search capabilities via the Serper.dev API. It offers clear, simple, and user-friendly answers, with the ability to handle both medical and general queries through an interactive terminal interface.

## Features
- **Health-Focused Responses**: Provides accurate and helpful answers to health-related questions using context from web searches.
- **General Query Support**: Answers non-health-related questions with relevant information.
- **Real-Time Web Search**: Utilizes the Serper.dev API to fetch up-to-date information from the web.
- **Interactive Terminal Interface**: Allows users to interact with the chatbot via a command-line interface.
- **Powered by Groq**: Leverages the `llama-3.3-70b-versatile` model for natural language processing.

## Prerequisites
To run talkDOC, you need the following:
- Python 3.8 or higher
- A `.env` file with the following API keys:
  - `SERPER_API_KEY`: Obtain from [Serper.dev](https://serper.dev/)
  - `GROQ_API_KEY`: Obtain from [xAI](https://x.ai/api)
- Internet connection for API requests

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
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root with the following content:
   ```
   SERPER_API_KEY=your-serper-api-key
   GROQ_API_KEY=your-groq-api-key
   ```

## Usage
1. **Run the Chatbot**:
   ```bash
   python talkDOC.py
   ```
   This starts the interactive terminal interface.

2. **Interact with talkDOC**:
   - Type your question (e.g., "What are the symptoms of a cold?").
   - The chatbot will fetch relevant context from the web and provide a clear, helpful response.
   - To exit, type `exit` or `quit`.

3. **Example Interaction**:
   ```
   🤖 AI Doctor Chatbot (type 'exit' to quit)

   You: What are the symptoms of a cold?
   AI Doctor: Common symptoms of a cold include a runny or stuffy nose, sore throat, cough, sneezing, and mild fatigue. Symptoms usually resolve within 7-10 days. If symptoms persist or worsen, consult a healthcare professional.

   You: exit
   👋 Goodbye!
   ```

## Code Structure
- **talkDOC.py**: Main script containing the chatbot logic.
  - `retrieve_docs`: Fetches top 3 search result snippets using Serper.dev API.
  - `get_context`: Combines snippets into a context string.
  - `answer_query`: Uses the Groq LLM to generate a response based on context and user query.
  - Interactive loop for terminal-based chat.
- **.env**: Stores API keys for Serper.dev and Groq.
- **requirements.txt**: Lists required Python packages.

## Dependencies
- `requests`: For making HTTP requests to the Serper.dev API.
- `python-dotenv`: For loading environment variables from a `.env` file.
- `langchain-groq`: For integrating with the Groq language model.
- `langchain-core`: For building the prompt and chain logic.

## Limitations
- **Health Advice**: talkDOC is not a substitute for professional medical advice. Always consult a licensed healthcare provider for medical concerns.
- **API Dependency**: Requires valid API keys and an internet connection.
- **Search Results**: Limited to the top 3 snippets from Serper.dev, which may not always cover all relevant information.

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