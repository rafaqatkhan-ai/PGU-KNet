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
        # Use a property instead of directly assigning dtype
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

# UI
st.title("🧠 CT Kidney Image Classifier")
st.write("Upload a CT kidney image, and the model will predict its class.")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image('your_image.png', use_column_width=True)


    # Preprocess
    img_size = (224, 224)
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### ✅ **Prediction:** `{predicted_class}`")
    st.markdown(f"### 🔍 **Confidence:** `{confidence:.2%}`")
