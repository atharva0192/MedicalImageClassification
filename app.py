import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('MyCNN.h5')

# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((256, 256))  # Resize image to match model input size
    img_array = np.array(img)  # Convert image to NumPy array
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    y_predict= []
    for ele in predictions:
    if ele>=0.5 :
      y_predict.append(1)
    else:
      y_predict.append(0)

    return y_predict

# Streamlit app code
st.title("Deep Learning Model Deployment")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Perform prediction
    result = predict(uploaded_file)

    # Display the prediction results
    st.write("Prediction Results:")
    st.write(result)
