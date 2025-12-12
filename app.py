# app.py — Interface moderne
import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration 
MODEL_PATH = "apple_disease_model1.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Apple Scab", "Black Rot", "Cedar Apple Rust"]

st.set_page_config(
    page_title="Diagnostic maladies du pommier",
    layout="centered",
    page_icon="🍏"
)

#  CSS STYLE 
st.markdown("""
<style>
    /* Fond bleu ciel */
    .reportview-container, .main, .block-container {
        background-color: #87CEEB;  /* Sky Blue */
    }

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 10px;
    }

    .sub {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    .upload-box {
        border: 2px dashed #8BC34A;
        padding: 20px;
        border-radius: 15px;
        background-color: #FAFFFA;
    }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        color: white;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


#  Charger modèle 
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return tf.keras.models.load_model(path, compile=False)


# Prétraitement 
def preprocess_pil_image(pil_img: Image.Image, target_size=IMG_SIZE):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img = pil_img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


# Prédiction 
def predict_image(model, pil_img: Image.Image):
    x = preprocess_pil_image(pil_img)
    preds = model.predict(x)

    probs = tf.nn.softmax(preds[0]).numpy()
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


#  INTERFACE 
st.markdown("<h1 class='main-title'>🍏 Diagnostiqueur de maladies du pommier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Importe une image pour identifier la maladie présente sur la feuille.</p>", unsafe_allow_html=True)


# Zone d’upload avec style
with st.container():
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image importée", width=700)

    model = load_model(MODEL_PATH)

    if st.button("🔍 Lancer le diagnostic"):
        with st.spinner("Analyse en cours..."):
            idx, conf, probs = predict_image(model, image)

        label = CLASS_NAMES[idx]
        pct = round(conf * 100, 2)

        # Couleur selon classe
        color_map = {
            "Apple Scab": "#FBC02D",
            "Black Rot": "#D32F2F",
            "Cedar Apple Rust": "#388E3C"
        }

        st.markdown(
            f"<div class='result-card' style='background-color:{color_map[label]}'>"
            f"Maladie détectée : {label}<br>"
            f"Probabilité : {pct}%"
            f"</div>",
            unsafe_allow_html=True
        )
