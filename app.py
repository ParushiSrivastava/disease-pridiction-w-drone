import streamlit as st
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = load_model("plant_disease_efficientnet.keras")

    with open("class_indices.json") as f:
        idx_to_class = json.load(f)

    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return model, idx_to_class


def fix_image_array(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # White balance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])

    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1

    img = cv2.cvtColor(np.clip(lab, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

    # Bilateral filter
    img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    return img


def estimate_severity_overlay(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    bg_mask = ((v < 40) | (v > 240) | (s < 30))

    green_mask = (
        (h >= 35) & (h <= 85) &
        (s >= 40) & (v >= 40) &
        ~bg_mask
    )

    brown_mask = (
        (h >= 8) & (h <= 30) &
        (s >= 50) &
        (v >= 50) &
        ~bg_mask
    )

    dark_brown_mask = (
        ((h <= 8) | (h >= 165)) &
        (s >= 50) &
        (v >= 40) & (v <= 180) &
        ~bg_mask
    )

    diseased_mask = brown_mask | dark_brown_mask

    n_green = int(np.sum(green_mask))
    n_diseased = int(np.sum(diseased_mask))
    total_leaf = max(n_green + n_diseased, 1)

    pct = round((n_diseased / total_leaf) * 100, 1)

    if n_diseased == 0:
        label = "Healthy"
    elif pct < 5:
        label = "Healthy"
    elif pct < 20:
        label = "Mild"
    elif pct < 50:
        label = "Moderate"
    else:
        label = "Severe"

    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    overlay[diseased_mask] = [232, 174, 112]  # light orange

    return pct, label, overlay


def predict_image(model, idx_to_class, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = preprocess_input(img_rgb.astype("float32"))
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = idx_to_class[class_index]

    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant = class_name
        disease = "Unknown"

    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(idx_to_class[int(i)], float(preds[int(i)])) for i in top3_idx]

    return plant, disease, confidence, top3


def clean_label(label):
    return label.replace("___", " — ").replace("_", " ")


model, idx_to_class = load_assets()

st.title(" Plant Disease Detection")
st.write("Upload or capture a leaf image to see enhancement, severity overlay, and disease prediction.")

mode = st.radio("Choose input method", ["Upload Image", "Take Photo"])

uploaded_file = None
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, 1)

    if original_bgr is None:
        st.error("Could not read the image.")
    else:
        original_rgb = cv2.cvtColor(cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        fixed_bgr = fix_image_array(original_bgr)
        fixed_rgb = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)

        severity_pct, severity_label, severity_overlay = estimate_severity_overlay(fixed_bgr)
        plant, disease, confidence, top3 = predict_image(model, idx_to_class, fixed_bgr)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(original_rgb, use_container_width=True)

            st.subheader("Enhanced Image")
            st.image(fixed_rgb, use_container_width=True)

        with col2:
            st.subheader(f"Severity: {severity_pct}% — {severity_label}")
            st.image(severity_overlay, use_container_width=True)

            st.subheader("Prediction Result")
            st.write(f"**Plant:** {plant}")
            st.write(f"**Disease:** {'Healthy' if disease.lower() == 'healthy' else disease.replace('_', ' ')}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

            st.subheader("Top 3 Predictions")
            for name, score in top3:
                st.write(f"- {clean_label(name)} : {score * 100:.2f}%")
else:
    st.info("Upload or capture an image to start.")
