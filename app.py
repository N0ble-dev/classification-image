import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
try:
    model = load_model('image_classification_model.keras')
except Exception as e:
    st.error(f"Failed to load model. Ensure the model file is available: {e}")
    st.stop()

# Define the class names (CIFAR-10 categories)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# App title
st.title("Image Classification App")
st.write("Upload an image to classify it into one of the CIFAR-10 categories.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False)

        # Preprocess the image for the model
        st.write("Preprocessing the image...")
        model_input_image = image.resize((32, 32))  # Resize to match model input size
        image_array = img_to_array(model_input_image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Check input shape
        st.write("Prepared image shape for model:", image_array.shape)

        # Predict the class
        predictions = model.predict(image_array)
        st.write("Raw model predictions:", predictions)

        # Extract the predicted class and confidence score
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        # Display the results
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence Score: {confidence:.2f}%")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

else:
    st.info("Please upload an image to begin.")
