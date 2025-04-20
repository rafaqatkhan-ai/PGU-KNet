import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom layer definition (needed for model loading)
class PositionalGatingUnit(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(PositionalGatingUnit, self).__init__(**kwargs)
        self.channels = channels
        self.conv = tf.keras.layers.Conv2D(channels, (1, 1), activation='sigmoid')

    def call(self, x):
        pos = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        gate = self.conv(pos)
        return x * gate

    def get_config(self):
        config = super(PositionalGatingUnit, self).get_config()
        config.update({"channels": self.channels})
        return config

# Load model with custom object
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Models/KidneyModel_Lightweight.h5", 
                                      custom_objects={'PositionalGatingUnit': PositionalGatingUnit})

model = load_model()
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']  # Update if necessary

st.title("CT Kidney Image Classifier")

uploaded_file = st.file_uploader("Upload a CT Kidney Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_size = (224, 224)
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"**Prediction:** {predicted_class} ({confidence:.2%})")
