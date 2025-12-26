import os
import tempfile
import base64
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from langdetect import detect
from deep_translator import GoogleTranslator
from backend import answer_query, is_medical_question, analyze_image_with_llama
from dotenv import load_dotenv

# =================== Load API Keys ===================
load_dotenv()


# =================== Helper Functions ===================




# =================== Streamlit UI ===================
st.set_page_config(page_title="talkDOC - Medical Assistant", page_icon="🩺", layout="centered")
st.title("🩺 talkDOC - Medical Assistant")
st.markdown("Upload a **medical or skin image**, or ask using **voice** or **text**. talkDOC can only respond to medical-related questions.")

# ----------- Image Upload -----------
image_file = st.file_uploader("📸 Upload medical/skin image:", type=["jpg", "jpeg", "png"])
image_question = st.text_input("Optional: Ask a question about this image:")

if st.button("Analyze Image"):
    if not image_file:
        st.warning("⚠️ Please upload an image first.")
    else:
        with st.spinner("⏳ Analyzing image..."):
            # Step 1: Vision model description
            image_description = analyze_image_with_llama(image_file.read(), image_question)
            st.markdown(f"**Description:** {image_description}")

            # Step 2: Check if description is medical
            if not is_medical_question(image_description):
                st.markdown("⚠️ Sorry, I can only answer medical-related questions.")
            else:
                # Step 3: Send to backend for medical reasoning
                detailed_answer = answer_query(image_description)
                st.markdown(f"**Medical Assistant Response:** {detailed_answer}")

# ----------- Voice Input -----------
if "voice_file" not in st.session_state:
    st.session_state.voice_file = None

audio_file = st.audio_input("🎙️ Record your health question:")
if audio_file is not None:
    st.session_state.voice_file = audio_file
    st.success("✅ Voice recorded. Press **Send Voice** to get an answer.")

if st.button("Send Voice"):
    if st.session_state.voice_file is None:
        st.warning("⚠️ Please record your question first.")
    else:
        try:
            with st.spinner("⏳ Processing voice..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(st.session_state.voice_file.getbuffer())
                    wav_path = tmp_wav.name

                # Convert to WAV if not already
                if not wav_path.endswith(".wav"):
                    sound = AudioSegment.from_file(wav_path)
                    tmp_wav_conv = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    wav_path = tmp_wav_conv.name
                    sound.export(wav_path, format="wav")

                # Speech recognition
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    user_text = recognizer.recognize_google(audio_data)

                # Language detection and translation
                lang = detect(user_text)
                user_text_en = GoogleTranslator(source="bn", target="en").translate(user_text) if lang == "bn" else user_text

                # Medical classification
                if not is_medical_question(user_text_en):
                    ai_response = "Sorry, I can only answer medical-related questions."
                else:
                    ai_response_en = answer_query(user_text_en)
                    ai_response = GoogleTranslator(source="en", target="bn").translate(ai_response_en) if lang == "bn" else ai_response_en

                st.markdown(f"**🗣️ You said:** {user_text}")
                st.markdown(f"**🤖 AI Doctor:** {ai_response}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ----------- Text Input -----------
user_text_input = st.text_input("💬 Or type your question here:", "")
if st.button("Send Text"):
    if not user_text_input.strip():
        st.warning("⚠️ Please type your question first.")
    else:
        try:
            with st.spinner("⏳ Processing text..."):
                lang = detect(user_text_input)
                user_text_en = GoogleTranslator(source="bn", target="en").translate(user_text_input) if lang == "bn" else user_text_input

                if not is_medical_question(user_text_en):
                    ai_response = "Sorry, I can only answer medical-related questions."
                else:
                    ai_response_en = answer_query(user_text_en)
                    ai_response = GoogleTranslator(source="en", target="bn").translate(ai_response_en) if lang == "bn" else ai_response_en

                st.markdown(f"**🗣️ You asked:** {user_text_input}")
                st.markdown(f"**🤖 AI Doctor:** {ai_response}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
