import streamlit as st
import numpy as np
from PIL import Image
import pickle
from keras.models import load_model
import tensorflow as tf

# -------------------------------
# Load trained CNN model
# -------------------------------
try:
    model = load_model("plant_disease_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load class names
try:
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
except Exception as e:
    st.error(f"Error loading class names: {e}")
    class_names = None

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Predictor",
    layout="centered"
)

st.title("🌱 Plant Disease Predictor")
st.write("Upload a plant leaf image to predict the disease.")

# -------------------------------
# Image Preprocessing (CNN)
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))        # SAME as training
    image = np.array(image)

    # Ensure RGB (3 channels)
    if image.shape[-1] != 3:
        image = image[:, :, :3]

    image = image / 255.0                   # Normalize
    image = np.expand_dims(image, axis=0)   # (1, 224, 224, 3)
    return image

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        if model is None or class_names is None:
            st.error("Model or class names failed to load. Please check your files.")
        else:
            img_array = preprocess_image(image)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_label = class_names[predicted_index]
            confidence = np.max(prediction) * 100

            st.success(f"🦠 Disease Detected: **{predicted_label}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")
