import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Layer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for Unknown layer: 'Cast'
class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.dtype = tf.dtypes.as_dtype(dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self.dtype.name})
        return config

# Add any other custom layers if needed
custom_objects = {
    'Cast': Cast,
    # Add other custom layers here if your model requires them
}

# Cache model loading with better error handling
@st.cache_resource
def load_model_cached(path):
    try:
        logger.info(f"Loading model from {path}")
        model = tf.keras.models.load_model(
            path, 
            compile=False, 
            custom_objects=custom_objects
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None

# Load models with error handling
try:
    resnet_model = load_model_cached('Models/KidneyModel_Lightweight.h5')
    efficientnet_model = resnet_model  # Using same model for both as in original code
    
    if resnet_model is None:
        st.error("Failed to load the model. Please check the model file.")
        st.stop()
except Exception as e:
    st.error(f"An error occurred during model loading: {str(e)}")
    st.stop()

# Class labels
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Page styling
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
        color: white;
    }
    .error-box {
        border: 2px solid #e74c3c;
        background-color: #1f1e1c;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="title">Kidney Disease Detection</h1>', unsafe_allow_html=True)

# Model selection
model_option = st.selectbox("Select a Model:", ["ResNet", "EfficientNet"])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image preprocessing
def preprocess_image(image):
    try:
        image = np.array(image)
        # Handle grayscale images
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        # Handle RGBA images
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        st.error(f"Error processing image: {str(e)}")
        return None

# Prediction function
def make_prediction(image, model):
    try:
        pred = model.predict(image)
        label = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100
        return label, confidence
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, None

# Main workflow
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        processed_image = preprocess_image(image)
        if processed_image is None:
            st.stop()
        
        selected_model = resnet_model if model_option == "ResNet" else efficientnet_model
        
        with st.spinner('Making prediction...'):
            label, confidence = make_prediction(processed_image, selected_model)
        
        if label and confidence:
            st.markdown(f"""
            <div class="prediction-box">
                <strong>Prediction:</strong> {label} with confidence <strong>{confidence:.2f}%</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
                <strong>Error:</strong> Could not make prediction. Please try another image.
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown('<div class="footer">© 2025 Kidney Disease Detection App</div>', unsafe_allow_html=True)
