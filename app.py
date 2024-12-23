import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('image_classification_model.keras')

# Define the class names (based on CIFAR-10 in this example)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# App title
st.title("Image Classification App")

# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Display the uploaded image in its original resolution
        st.image(image, caption="Uploaded Image", use_column_width=False)

        # Prepare a resized copy for the model input
        model_input_image = image.resize((32, 32))  # Resize for model input
        image_array = img_to_array(
            model_input_image) / 255.0  # Scale pixel values
        image_array = np.expand_dims(
            image_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        # Display the result
        st.write("Prediction:", predicted_class)
        st.write("Confidence Score:", f"{confidence:.2f}%")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
