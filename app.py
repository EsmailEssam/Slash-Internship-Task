import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

os.chdir(r"D:\Programing\Projects\Slash AI Intern")


model = load_model("Product_Classifier_model.h5")

class_labels_dict = {0: "Fashion", 1: "Accessories", 2: "Beauty", 3: "Home"}

st.title("Product Image Classifier")
st.text("Upload Image")

uploaded_img = st.file_uploader("Choose an image...", type="jpg")
if uploaded_img is not None:
    img = image.load_img(uploaded_img)
    st.image(img)

    if st.button("Predict"):
        x = image.smart_resize(img, (224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize pixel values
        pred = model.predict(x)
        predicted_class = np.argmax(pred)
        predicted_label = class_labels_dict[predicted_class]
        st.write(f"This item belongs to {predicted_label} category.")
