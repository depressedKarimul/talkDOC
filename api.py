from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend import answer_query, is_medical_question, analyze_image_with_llama, GROQ_API_KEY
import uvicorn
import os

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "TalkDOC API is running"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_message = request.message
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Optional: Check if medical related (can be disabled if we trust the user context or want general chat)
    if not is_medical_question(user_message):
       return {"response": "Sorry, I can only answer medical-related questions."}

    try:
        response = answer_query(user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...), question: Optional[str] = Form(None)):
    try:
        contents = await file.read()
        description = analyze_image_with_llama(contents, question)
        
        if not is_medical_question(description):
             return {
                 "description": description,
                 "response": "The image does not appear to be medical-related, so I cannot provide specific medical advice."
             }
        
        medical_response = answer_query(description)
        return {
            "description": description,
            "response": medical_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
