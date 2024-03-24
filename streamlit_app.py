import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.title("BREAST ABNORMALITY CLASSIFICATION")

# Load the saved model
model = tf.keras.models.load_model('abnormality.h5')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to RGB (if it's grayscale)
    image = image.convert("RGB")

    # Preprocess the image
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize the image data
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)

    # Decode the predictions
    class_names = ['calcification', 'mass']
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediction: {predicted_class}")
