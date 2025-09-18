import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
from tensorflow.keras.models import load_model as keras_load_model

# --- Konfigurasi ---
FILE_ID = "1J9KWogge6tfOCUBQJQg4CriobJOC32-y"
MODEL_PATH = "freshness_best_model.keras"
IMG_SIZE = (224, 224)

# --- Load Model ---
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return keras_load_model(MODEL_PATH)

model = get_model()

# --- Preprocessing ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # pastikan 3 channel
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0  # normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

# --- Streamlit UI ---
st.set_page_config(page_title="Freshness Detection", page_icon="🥗", layout="centered")

st.title("🥗 Food Freshness Detection")
st.write("Upload gambar atau foto langsung dengan kamera untuk mengetahui makanan itu **Segar** atau **Busuk**.")

tab1, tab2 = st.tabs(["📂 Upload Gambar", "📸 Pakai Kamera"])

image = None

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with tab2:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

# --- Prediction ---
if image is not None:
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    if prediction.shape[1] == 1:
        # Binary classification (Sigmoid)
        prob = prediction[0][0]
        label = "Segar" if prob < 0.5 else "Busuk"
        confidence = prob if label == "Busuk" else 1 - prob
    else:
        # Multi-class classification (Softmax)
        classes = ["Segar", "Busuk"]
        idx = np.argmax(prediction[0])
        label = classes[idx]
        confidence = prediction[0][idx]

    st.markdown(f"### 🥑 Prediction: **{label}** ({confidence*100:.2f}% confidence)")
