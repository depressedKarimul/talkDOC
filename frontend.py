import os
import tempfile
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS
from backend import answer_query  # তোমার backend থেকে answer_query import হবে

# Streamlit page configuration
st.set_page_config(page_title="talkDOC - Voice Doctor", page_icon="🩺", layout="centered")

# Title
st.title("🩺 talkDOC - Voice Doctor Assistant")
st.markdown("🎙️ Record your health-related question, then press **Send** to get an answer.")

# Session state for storing audio
if "voice_file" not in st.session_state:
    st.session_state.voice_file = None

# Voice input (record / upload)
audio_file = st.audio_input("Record your question here...")

if audio_file is not None:
    st.session_state.voice_file = audio_file  # store temporarily
    st.success("✅ Voice recorded. Now press **Send** to get answer.")

# Send button
if st.button("Send Voice"):
    if st.session_state.voice_file is None:
        st.warning("⚠️ Please record your question first.")
    else:
        try:
            with st.spinner("⏳ Generating answer... Please wait..."):
                # Save uploaded audio to temp WAV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(st.session_state.voice_file.getbuffer())
                    wav_path = tmp_wav.name

                # Convert to WAV if necessary
                if not wav_path.endswith(".wav"):
                    sound = AudioSegment.from_file(wav_path)
                    wav_path = wav_path.replace(".ogg", ".wav").replace(".mp3", ".wav")
                    sound.export(wav_path, format="wav")

                # Speech to Text
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    user_text = recognizer.recognize_google(audio_data)

                # Detect language
                lang = detect(user_text)

                # Translate to English if Bangla
                if lang == "bn":
                    user_text_en = GoogleTranslator(source="bn", target="en").translate(user_text)
                else:
                    user_text_en = user_text

                # Call backend AI Doctor
                ai_response_en = answer_query(user_text_en)

                # Translate back to Bangla if user spoke Bangla
                if lang == "bn":
                    ai_response = GoogleTranslator(source="en", target="bn").translate(ai_response_en)
                else:
                    ai_response = ai_response_en

                # Text to Speech (voice reply)
                tts = gTTS(ai_response, lang="bn" if lang == "bn" else "en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
                    tts.save(tmp_mp3.name)
                    st.audio(tmp_mp3.name, format="audio/mp3")

                # Show transcripts
                st.markdown(f"**🗣️ You said:** {user_text}")
                st.markdown(f"**🤖 AI Doctor:** {ai_response}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown(
    '<div style="text-align:center;color:#666;margin-top:20px;font-size:14px;">'
    'Powered by talkDOC | Not a substitute for professional medical advice'
    '</div>',
    unsafe_allow_html=True
)


# backend.py

