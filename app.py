import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("ultrasound_classifier.h5")

# Define class labels
class_labels = ["benign", "malignant", "normal"]  # Adjust as per your training order

# Streamlit UI
st.title("🧠 Fetal Ultrasound Report Analyzer")
st.subheader("Upload a fetal ultrasound image to detect abnormality (Normal / Benign / Malignant)")

uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Resize as per your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.markdown(f"### 🧪 Prediction: **{predicted_class.upper()}**")
    st.markdown(f"🧮 Confidence Scores: {dict(zip(class_labels, (prediction[0].round(2))*100))}")
