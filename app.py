import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Kidney CT Classifier", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "KidneyModel_Lightweight.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']  # Modify if class order is different

# Title
st.title("🧠 Kidney CT Scan Classifier")
st.write("Upload a CT scan image and let the model classify it into one of the four categories.")

# Image uploader
uploaded_file = st.file_uploader("Choose a CT scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)
    image_resized = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Result
    st.markdown(f"### 🩺 Predicted Class: `{predicted_class}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
