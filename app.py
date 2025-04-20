import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback

# Custom layer: PositionalGatingUnit
class PositionalGatingUnit(tf.keras.layers.Layer):
    def __init__(self, channels=32, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.conv = tf.keras.layers.Conv2D(channels, (1, 1), activation='sigmoid')

    def call(self, inputs):
        pos = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        gate = self.conv(pos)
        
        # Cast inputs and gate to float32 to avoid dtype mismatch
        inputs = tf.cast(inputs, tf.float32)
        gate = tf.cast(gate, tf.float32)
        
        return inputs * gate

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config

# Register dummy Cast layer for compatibility
class Cast(tf.keras.layers.Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self._dtype = tf.dtypes.as_dtype(dtype)

    def call(self, inputs):
        return tf.cast(inputs, self._dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self._dtype.name})
        return config

# Load the model safely with custom layers
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "KidneyModel_Lightweight.h5")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'PositionalGatingUnit': PositionalGatingUnit,
                'Cast': Cast
            }
        )
        return model
    except Exception as e:
        st.error("❌ Failed to load the model. Please check the file path or model content.")
        st.code(traceback.format_exc())
        st.stop()

# Load model
model = load_model()
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# UI - Customizing appearance
st.set_page_config(page_title="PGU-KNet CT Kidney Image Classifier", page_icon="🧠", layout="centered")

# Custom CSS to make the app more attractive
st.markdown("""
    <style>
    /* Background Gradient */
    .main {
        background: linear-gradient(to right, #4e8eff, #00c6ff);
        color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 36px;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .upload-text {
        font-size: 18px;
        color: #ffffff;
    }
    .result {
        font-size: 24px;
        color: #4CAF50;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
    .confidence {
        font-size: 20px;
        color: #FF5722;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .btn {
        background-color: #FF5722;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .btn:hover {
        background-color: #ff7043;
    }
    </style>
""", unsafe_allow_html=True)

# UI content
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🧠 PGU-KNet CT Kidney Image Classifier")
st.write("Upload a CT kidney image, and the model will predict its class.")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_size = (224, 224)
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Show progress bar while predicting
    with st.spinner('Classifying the image...'):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    # Show result in attractive format
    st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown(f'<p class="result">✅ **Prediction:** {predicted_class}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="confidence">🔍 **Confidence:** {confidence:.2%}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Closing the div of main layout
st.markdown('</div>', unsafe_allow_html=True)
