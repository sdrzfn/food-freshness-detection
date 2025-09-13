import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load Model
model = tf.keras.models.load_model("freshness_best_model.keras")


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
st.write("Upload or capture a food image to check if it's **Fresh** or **Rotten**.")

tab1, tab2 = st.tabs(["📂 Upload Image", "📸 Use Camera"])

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
        label = "Fresh" if prob < 0.5 else "Rotten"
        confidence = prob if label == "Rotten" else 1 - prob
    else:
        classes = ["Fresh", "Rotten"]
        idx = np.argmax(prediction[0])
        label = classes[idx]
        confidence = prediction[0][idx]

    st.markdown(f"### 🥑 Prediction: **{label}** ({confidence*100:.2f}% confidence)")
