import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

model = tf.keras.models.load_model('emotion_detection_model.h5')

def prediction(file):
    img = Image.open(file).convert('RGB')
    img = img.resize((48,48))
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)

    classes = model.predict(x)
    idx = np.argmax(classes)
    clas = ['mawrah', 'eneg', 'takut', 'senang', 'sedih', 'terkejut', 'netral']
    return clas[idx]

st.title("Emotion Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = prediction(uploaded_file)
    st.write(f"Prediction: {label}")
    