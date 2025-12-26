# talkDOC - AI Doctor Chatbot 🩺

## Overview
**talkDOC** is an advanced AI-powered health assistant designed to provide accurate, fact-checked, and helpful responses to medical inquiries. It leverages **Groq**'s high-performance LLMs (Llama 3, Qwen) and real-time Google Search results via **Serper.dev** to ensure up-to-date information.

The ecosystem consists of:
1.  **Backend Logic (`backend.py`)**: Core AI processing, vector database (FAISS) integration, and answer refinement.
2.  **API Server (`api.py`)**: A **FastAPI** server that exposes the chatbot's functionality to external clients (like the mobile app).
3.  **App Interface (`frontend.py`)**: A **Streamlit** web application for easy interaction via text, voice, and image analysis.
4.  **Mobile App (`mobile/`)**: A **React Native (Expo)** mobile application for on-the-go access.

---

## 🚀 Features

### Core AI Capabilities
-   **Context-Aware Responses**: Uses a **FAISS vector store** created from a comprehensive disease-diagnosis dataset ("sajjadhadi/disease-diagnosis-dataset") to ground answers in medical data.
-   **Dual-Model Refinement**:
    -   **Drafting**: `llama-3.3-70b-versatile` generates an initial answer based on knowledge base context.
    -   **Refining**: `qwen/qwen3-32b` refines the answer using **real-time Google Search snippets** for verification and additional details.
-   **Medical Classification**: A dedicated model (`llama-3.1-8b-instant`) classifies user inputs to ensure the bot **only** responds to health-related queries.
-   **Image Analysis**: Uses `meta-llama/llama-4-scout-17b-16e-instruct` to analyze medical images (e.g., skin conditions), generate descriptions, and provide medical insights.

### Interface-Specific Features
-   **Streamlit Web App**:
    -   Voice input with speech-to-text (Google API).
    -   Automatic language detection and translation (Bangla <-> English).
    -   Image upload and analysis.
-   **Mobile App**:
    -   Identify skin conditions via camera or gallery upload.
    -   Chat interface with the AI doctor.
    -   Real-time server communication.

---

## 🛠️ Installation & Setup

### Prerequisites
-   Python 3.8+
-   Node.js & npm (for Mobile App)
-   Expo Go app on your phone (for testing Mobile App)
-   API Keys:
    -   **Groq API Key**: [Get it here](https://console.groq.com)
    -   **Serper.dev API Key**: [Get it here](https://serper.dev)

### 1. Clone the Repository
```bash
git clone https://github.com/depressedKarimul/talkDOC.git
cd talkDOC
```

### 2. Backend Setup (Python)
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

**Initialize the Knowledge Base:**
Run this script once to download the dataset and create the FAISS vector index (`faiss_symptom_disease`).
```bash
python memory.py
```

### 3. Run the Application

#### Option A: CLI Chatbot
Run the terminal-based interactive chat:
```bash
python backend.py
```

#### Option B: Streamlit Web Dashboard
Launch the web interface (accessible at `http://localhost:8501`):
```bash
streamlit run frontend.py
```

#### Option C: API Server & Mobile App
**Step 1: Start the API Server**
The mobile app communicates with this FastAPI server.
```bash
python api.py
```
*Server runs on `http://0.0.0.0:8000`.*

**Step 2: Start the Mobile App**
Navigate to the mobile directory and install dependencies:
```bash
cd mobile
npm install
```

**Step 3: Configure API Endpoint**
Open `mobile/App.js` and update the `API_URL` to your computer's local IP address (so your phone can reach it):
```javascript
// Example in mobile/App.js
const API_URL = 'http://192.168.1.X:8000'; // Replace with your local IP
```

**Step 4: Run Expo**
```bash
npx expo start
```
Scan the QR code with the **Expo Go** app on your Android/iOS device.

---

## 📂 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `backend.py` | Main logic: loads models, handles FAISS retrieval, Google Search, and answer generation. |
| `api.py` | FastAPI server providing `/chat` and `/analyze-image` endpoints for the mobile app. |
| `frontend.py` | Streamlit UI code for the web dashboard. |
| `memory.py` | ETL script: downloads dataset, embeds text, builds FAISS index. |
| `mobile/` | React Native (Expo) project source code. |
| `data/` | Directory for datasets. |
| `faiss_symptom_disease/` | Generated vector store folder. |

---

## ⚠️ Disclaimer
**talkDOC is an AI tool and NOT a real doctor.**
-   It provides information based on medical datasets and web searches.
-   **Always consult a certified medical professional** for diagnosis and treatment.
-   In case of emergency, contact your local emergency services immediately.

---

## 🤝 Contributing
Contributions are welcome!
1.  Fork the repo.
2.  Create a feature branch (`git checkout -b feature-new`).
3.  Commit changes (`git commit -m 'Add new feature'`).
4.  Push to branch (`git push origin feature-new`).
5.  Open a Pull Request.

## 📄 License
This project is licensed under the MIT License.