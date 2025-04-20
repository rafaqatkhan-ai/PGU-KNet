import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Custom Layer Definition ---
class PositionalGatingUnit(tf.keras.layers.Layer):
    def __init__(self, channels=32, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.conv = tf.keras.layers.Conv2D(channels, (1, 1), activation='sigmoid')

    def call(self, inputs):
        pos = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        gate = self.conv(pos)
        return inputs * gate

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config

# --- Load Model with Correct Path ---
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "KidneyModel_Lightweight.h5")
    try:
        model = tf.keras.models.load_model(model_path, 
                                           custom_objects={'PositionalGatingUnit': PositionalGatingUnit})
        return model
    except Exception as e:
        st.error("❌ Failed to load the model. Please check the file path or model content.")
        st.stop()

model = load_model()
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']  # Modify if your model uses different labels

# --- Streamlit App UI ---
st.title("🧠 CT Kidney Image Classifier")

uploaded_file = st.file_uploader("📤 Upload a CT Kidney Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # --- Preprocess the image ---
    img_size = (224, 224)
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Predict ---
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### ✅ **Prediction:** `{predicted_class}` with confidence **{confidence:.2%}**")
