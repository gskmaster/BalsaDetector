import streamlit as st
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
import tempfile

# Initialize Roboflow
rf = Roboflow(api_key="9T8zDm97SGYJGfcidwAX")
project = rf.workspace().project("wood-defect-detector")
model = project.version(1).model

st.set_page_config(page_title="Wood Block Scanner", layout="centered")
st.title("🪵 Wood Block Scanner & Analyzer")

uploaded_file = st.file_uploader("📸 Take or Upload a Wood Block Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_image_path = tmp_file.name

    st.image(temp_image_path, caption="📷 Uploaded Image", use_column_width=True)

    # Inference with Roboflow
    prediction = model.predict(temp_image_path).json()
    st.subheader("✅ Defect Detection Results")

    for obj in prediction['predictions']:
        st.write(f"🟡 Detected: {obj['class']} | Confidence: {round(obj['confidence'] * 100, 2)}%")

    # Save and display annotated result
    model.predict(temp_image_path).save("annotated.jpg")
    st.image("annotated.jpg", caption="🔍 Highlighted Results", use_column_width=True)