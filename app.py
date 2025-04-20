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
st.set_page_config(page_title="CT Kidney Image Classifier", page_icon="🧠", layout="centered")

# Custom CSS to make the app more attractive
st.markdown("""
    <style>
    body {
        background-color: #f0f0f5;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #2c3e50;
        margin-top: -50px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 50px;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #bdc3c7;
        margin-top: 50px;
    }
    .uploaded-image {
        border: 2px solid #2c3e50;
        border-radius: 10px;
        margin: 20px 0;
    }
    .prediction-box {
        border: 2px solid #27ae60;
        background-color: #1f1e1c;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #ffffff;
    }
    .btn {
        background-color: #2ecc71;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .btn:hover {
        background-color: #27ae60;
    }
    </style>
""", unsafe_allow_html=True)

# UI content
st.markdown('<div class="title">🧠 CT Kidney Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a CT kidney image, and the model will predict its class.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True, class_="uploaded-image")

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
    st.markdown(f"""
        <div class="prediction-box">
            <strong>Prediction:</strong> {predicted_class} with confidence <strong>{confidence:.0f}%</strong>
        </div>
    """, unsafe_allow_html=True)

# Footer section
st.markdown('<div class="footer">© 2025 Kidney Disease Detection App</div>', unsafe_allow_html=True)
