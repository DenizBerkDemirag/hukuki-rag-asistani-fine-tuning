"""
Türk Hukuku Asistanı — Basit Streamlit Arayüzü.
Yalnızca eğitilmiş LoRA modeli mevcutsa çalışır.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from src.inference import is_model_ready, load_model, generate_stream

st.set_page_config(page_title="Türk Hukuku Asistanı", page_icon="⚖️")
st.title("⚖️ Türk Hukuku Asistanı")

# Eğitilmiş model yoksa uygulamayı durdur
if not is_model_ready():
    st.error(
        "Eğitilmiş model bulunamadı.\n\n"
        f"Beklenen konum: `{config.OUTPUT_DIR}`\n\n"
        "Önce modeli eğitin: `python src/train.py`"
    )
    st.stop()


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model():
    return load_model(allow_base_fallback=False)


model, tokenizer = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hukuki sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = st.write_stream(
            generate_stream(
                question=prompt,
                chat_history=st.session_state.messages[:-1],
                model=model,
                tokenizer=tokenizer,
            )
        )
    st.session_state.messages.append({"role": "assistant", "content": answer})