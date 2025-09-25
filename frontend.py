

import os
import tempfile
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS
from backend import answer_query
from groq import Groq

# ========== Groq setup ==========
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_medical_question(question: str) -> bool:
    prompt = f"""
    You are a classifier. Determine if the following question is related to
    health, medicine, diseases, symptoms, treatments, or healthcare.
    Answer only with YES or NO.

    Question: "{question}"
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # updated model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )
    result = response.choices[0].message.content.strip().upper()
    return result.startswith("YES")

# ========== Streamlit UI ==========

st.set_page_config(page_title="talkDOC - Voice Doctor", page_icon="🩺", layout="centered")
st.title("🩺 talkDOC - Voice Doctor Assistant")
st.markdown("🎙️ Record your health-related question or type it below, then press **Send** to get an answer.")

if "voice_file" not in st.session_state:
    st.session_state.voice_file = None

audio_file = st.audio_input("Record your question here...")
if audio_file is not None:
    st.session_state.voice_file = audio_file
    st.success("✅ Voice recorded. Now press **Send Voice** to get answer.")

if st.button("Send Voice"):
    if st.session_state.voice_file is None:
        st.warning("⚠️ Please record your question first.")
    else:
        try:
            with st.spinner("⏳ Processing voice..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(st.session_state.voice_file.getbuffer())
                    wav_path = tmp_wav.name

                if not wav_path.endswith(".wav"):
                    sound = AudioSegment.from_file(wav_path)
                    tmp_wav_conv = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    wav_path = tmp_wav_conv.name
                    sound.export(wav_path, format="wav")

                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    user_text = recognizer.recognize_google(audio_data)

                lang = detect(user_text)
                user_text_en = GoogleTranslator(source="bn", target="en").translate(user_text) if lang == "bn" else user_text

                if not is_medical_question(user_text_en):
                    ai_response = "I’m sorry, I can only provide medical and health-related suggestions. I won’t be able to answer questions outside this topic."
                else:
                    ai_response_en = answer_query(user_text_en)
                    ai_response = GoogleTranslator(source="en", target="bn").translate(ai_response_en) if lang == "bn" else ai_response_en

                tts = gTTS(ai_response, lang="bn" if lang == "bn" else "en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
                    tts.save(tmp_mp3.name)
                    st.audio(tmp_mp3.name, format="audio/mp3")

                st.markdown(f"**🗣️ You said:** {user_text}")
                st.markdown(f"**🤖 AI Doctor:** {ai_response}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

user_text_input = st.text_input("Or type your question here:", "")

if st.button("Send Text"):
    if not user_text_input.strip():
        st.warning("⚠️ Please type your question first.")
    else:
        try:
            with st.spinner("⏳ Processing text..."):
                lang = detect(user_text_input)
                user_text_en = GoogleTranslator(source="bn", target="en").translate(user_text_input) if lang == "bn" else user_text_input

                if not is_medical_question(user_text_en):
                    ai_response = "Sorry, ami bairer question er answer dite parbo na."
                else:
                    ai_response_en = answer_query(user_text_en)
                    ai_response = GoogleTranslator(source="en", target="bn").translate(ai_response_en) if lang == "bn" else ai_response_en

                tts = gTTS(ai_response, lang="bn" if lang == "bn" else "en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
                    tts.save(tmp_mp3.name)
                    st.audio(tmp_mp3.name, format="audio/mp3")

                st.markdown(f"**🗣️ You asked:** {user_text_input}")
                st.markdown(f"**🤖 AI Doctor:** {ai_response}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
