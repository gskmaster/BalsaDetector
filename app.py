import streamlit as st
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
import tempfile

# Set page config for mobile
st.set_page_config(
    page_title="Wood Block Scanner",
    layout="centered",
    page_icon="🪵"
)

st.title("🪵 Wood Block Scanner & Analyzer")

# Roboflow Initialization
rf = Roboflow(api_key="9T8zDm97SGYJGfcidwAX")
project = rf.workspace("gbi").project("balsa-defect-detector")
model = project.version(1).model

# --- Camera input for mobile
st.markdown("## 📸 Capture or Upload Wood Block Image")

use_camera = st.toggle("Use camera", value=True)

if use_camera:
    image_file = st.camera_input("Take a picture")
else:
    image_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if image_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_file.read())
        temp_image_path = tmp_file.name

    st.image(temp_image_path, caption="📷 Input Image", use_column_width=True)

    try:
        # Predict with Roboflow model
        prediction = model.predict(temp_image_path).json()
        st.subheader("✅ Defect Detection Results")

        for obj in prediction['predictions']:
            st.write(f"🟡 Detected: {obj['class']} | Confidence: {round(obj['confidence'] * 100, 2)}%")

        # Save and show annotated result
        model.predict(temp_image_path).save("annotated.jpg")
        st.image("annotated.jpg", caption="🔍 Annotated Result", use_column_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
