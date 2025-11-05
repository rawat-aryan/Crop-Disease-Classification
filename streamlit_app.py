import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_trained_model():
    model = load_model("Plant_disease_classifier.h5")
    return model

model = load_trained_model()

# -----------------------
# Class names (update as per your dataset)
# -----------------------
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(page_title="Crop Disease Classifier", layout="centered")

st.title(" Plant Disease Classification App")
st.write("Upload a leaf image to identify the disease using a MobileNetV3-based model.")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

    # Predict
    with st.spinner("Analyzing... "):
        preds = model.predict(img_array)
        predicted_idx = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

    # Show result
    st.success(f"**Prediction:** {CLASS_NAMES[predicted_idx]}")
    st.info(f" **Confidence:** {confidence*100:.2f}%")

    # Interpretation
    if "healthy" in CLASS_NAMES[predicted_idx].lower():
        st.balloons()
        st.write("🌱 The plant appears healthy. No immediate action required.")
    else:
        st.warning("Disease detected! It’s recommended to apply appropriate treatment or consult an agronomist.")
