import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import os
import base64
from PIL import Image

# ✅ Step 1: Download the model from Google Drive if not found
MODEL_URL = "https://drive.google.com/uc?id=13di0px10kBfKqgdaI6B8aIFeoHsAmVhb"
MODEL_PATH = "model.h5"

def download_model():
    """Downloads the model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... Please wait.")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.write("Model downloaded successfully! ✅")
        else:
            st.error("Error downloading model. Check the URL or try again.")
            return

download_model()

# ✅ Step 2: Load the trained model
try:
    model = load_model(MODEL_PATH)
    st.write("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ✅ Step 3: Define class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# ✅ Step 4: Set a background image
def set_background(image_path):
    with open(image_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1, h3, h4, p, label {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

set_background("background.jpg")  # Replace with a valid medical-themed image

# ✅ Step 5: Function to predict brain tumor
def predict_tumor(image):
    """Processes image and makes a prediction using the model."""
    image = image.resize((128, 128))  # Resize to match model input
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    
    if class_labels[predicted_class_index] == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index]}"
    
    return result, confidence_score

# ✅ Step 6: Streamlit UI
st.markdown(
    "<h1 style='text-align: center; color: white;'> Brain Tumor Detection</h1>",
    unsafe_allow_html=True,
)
st.write(
    "<p style='text-align: center; font-size: 18px; color: white;'>Upload an MRI scan to check for brain tumor presence.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    prediction, confidence = predict_tumor(image)
    
    st.markdown(
        f"<h3 style='color: white;'>Prediction: {prediction}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h4 style='color: white;'>Confidence: {confidence * 100:.2f}%</h4>",
        unsafe_allow_html=True,
    )
