import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
output = "freshness_best_model.tflite"
model = tf.keras.models.load_model(output)


# Preprocessing
IMG_SIZE = (224, 224)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array


# Streamlit UI
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

if image is not None:
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    if prediction.shape[1] == 1:
        prob = prediction[0][0]
        label = "Segar" if prob < 0.5 else "Busuk"
        confidence = prob if label == "Busuk" else 1 - prob
    else:
        classes = ["Segar", "Busuk"]
        idx = np.argmax(prediction[0])
        label = classes[idx]
        confidence = prediction[0][idx]

    st.markdown(f"### 🥑 Prediction: **{label}** ({confidence*100:.2f}% confidence)")
