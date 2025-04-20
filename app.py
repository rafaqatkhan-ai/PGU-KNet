import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback
import time  # Added this import to fix the error

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
st.set_page_config(
    page_title="PGU-KNet CT Kidney Image Classifier", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern attractive design
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Gradient header */
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Title styling */
    .title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Upload container */
    .upload-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    /* Upload button styling */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Prediction card */
    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin-top: 1rem;
    }
    
    /* Result text */
    .result {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    /* Confidence text */
    .confidence {
        font-size: 1.3rem;
        color: #7f8c8d;
    }
    
    /* Highlight text */
    .highlight {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .title {
            font-size: 2rem;
        }
        .subtitle {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="header">
        <div class="title">🧠 PGU-KNet CT Kidney Image Classifier</div>
        <div class="subtitle">Advanced deep learning model for classifying kidney CT scans into Cyst, Normal, Stone, or Tumor</div>
    </div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div class="upload-container">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">📁 Upload Your CT Scan</h3>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            # Open and display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("❌ Failed to process the image. Please try another file.")
            st.code(traceback.format_exc())
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        try:
            # Preprocess
            img_size = (224, 224)
            image = image.resize(img_size)
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Show progress bar while predicting
            with st.spinner('Analyzing the CT scan...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)  # Simulate processing time
                    progress_bar.progress(percent_complete + 1)
                
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

            # Show result in attractive format
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">🔍 Analysis Results</h3>
                    <div class="result">Predicted Condition: <span class="highlight">{predicted_class}</span></div>
                    <div class="confidence">Confidence Level: <span class="highlight">{confidence:.2%}</span></div>
                    
                    <div style="margin-top: 2rem;">
                        <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">Probability Distribution:</h4>
            """, unsafe_allow_html=True)
            
            # Show probability distribution as a bar chart
            prob_data = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
            st.bar_chart(prob_data, color="#667eea", height=250)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error("❌ An error occurred during prediction. Please try again.")
            st.code(traceback.format_exc())
    else:
        st.markdown("""
            <div style="background: white; border-radius: 15px; padding: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05); height: 100%; 
                     display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                <img src="https://cdn-icons-png.flaticon.com/512/3652/3652191.png" width="120" style="opacity: 0.7; margin-bottom: 1.5rem;">
                <h3 style="color: #2c3e50;">No Image Uploaded</h3>
                <p style="color: #7f8c8d;">Upload a kidney CT scan image to get started with the analysis.</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>PGU-KNet CT Kidney Classifier | Developed with TensorFlow and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Add some space at the bottom
st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
