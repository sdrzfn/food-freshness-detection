from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model("freshness_best_model.tflite")

@app.get("/")
async def root():
    return {"message": "Freshness Detection API is running!"}

@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Segar" if prediction < 0.5 else "Busuk"

    return {"label": label, "confidence": float(prediction)}
