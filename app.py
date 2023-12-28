import streamlit as st
import tensorflow as tf
import base64
import time
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('MyCNN.h5')

# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((256, 256))  # Resize the image to match the model input size
    img_array = np.array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    return predictions

# Streamlit app code
st.set_page_config(
    page_title="Classification of Medical X-Rays",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Classification of Medical X-Rays")
st.sidebar.write(
    "In recent years, the intersection of medical imaging and deep learning has witnessed unprecedented advancements, revolutionizing the landscape of healthcare. One notable application that has gained substantial attention is medical image classification using Convolutional Neural Networks (CNNs). As we embark on this project, we delve into the realm of leveraging cutting-edge deep learning techniques to augment traditional medical image analysis"
)

# Main content
st.title("Classification of Medical X-Rays")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image with border
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    # Perform prediction
    if st.button("Predict"):
        result = predict(uploaded_file)
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart = st.line_chart(np.random.randn(10, 2))
        
        for i in range(100):
            # Update progress bar.
            progress_bar.progress(i + 1)
        
            new_rows = np.random.randn(10, 2)
        
            # Update status text.
            status_text.text(
                f'Progress: %{i}')
        
            # Append data to the chart.
            chart.add_rows(new_rows)
        
            # Pretend we're doing some computation that takes time.
            time.sleep(0.03)
        
        status_text.text('Done!')
        
        # Display the prediction results
        st.write("Prediction Results:")
        prediction_label = "Normal" if result >= 0.5 else "Infected"
        if prediction_label == "Normal":
            st.balloons()
            st.success(f"The image is predicted as {prediction_label}")
        else :
            st.snow()
            st.warning(f'Warning : The image is predicted as {prediction_label}', icon="⚠️")

# Footer
st.markdown("---")
st.write("Developed by Atharva Chavan")
st.write("Copyright © 2023. All rights reserved.")
