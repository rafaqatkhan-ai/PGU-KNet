import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


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
    }
    </style>
""", unsafe_allow_html=True)


resnet_model = tf.keras.models.load_model('Models/KidneyModel_Lightweight.h5')
efficientnet_model = tf.keras.models.load_model('Models/KidneyModel_Lightweight.h5')


class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']


def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image


def make_prediction(image, model):
    pred = model.predict(image)
    label = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    return label, confidence


st.markdown('<h1 class="title">Kidney Disease Detection</h1>', unsafe_allow_html=True)



model_option = st.selectbox("Select a Model:", ["ResNet", "EfficientNet"])


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
  
    processed_image = preprocess_image(image)
    

    if model_option == "ResNet":
        selected_model = resnet_model
    else:
        selected_model = efficientnet_model
    

    label, confidence = make_prediction(processed_image, selected_model)
    

    st.markdown(f"""
    <div class="prediction-box">
        <strong>Prediction:</strong> {label} with confidence <strong>{confidence:.0f}%</strong>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">© 2025 Kidney Disease Detection App</div>', unsafe_allow_html=True)
