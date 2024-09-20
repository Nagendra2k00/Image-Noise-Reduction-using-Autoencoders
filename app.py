import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Enhanced MNIST Denoising and Classification", page_icon="🖼️")

# Custom CSS to improve UI
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border: 1px solid #d1d5db;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
    }
    .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #4a5568;
        padding: 20px;
        border-top: 1px solid #d1d5db;
        background-color: #f9fafb;
        margin-top: 20px;
    }
    .faq {
        background-color: #e2e8f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .faq h4 {
        color: #2c5282;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    autoencoder = load_model('autoencoder_model.h5')
    classifier = load_model('classifier_model.h5')
    denoise_and_classify = load_model('denoise_and_classify.h5')
    return autoencoder, classifier, denoise_and_classify

# Load data
@st.cache_data
def load_data():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32') / 255
    x_test = np.reshape(x_test, (10000, 784))
    return x_test, y_test

# Load models and data
autoencoder, classifier, denoise_and_classify = load_models()
x_test, y_test = load_data()

# Header
st.title("🖼️ MNIST Denoising and Classification App")
st.markdown("""
<p style='font-size: 20px; color: #4a5568;'>
Improve image classification with noise reduction using advanced deep learning techniques.
</p>
""", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("🎛️ Settings")
st.sidebar.markdown("""
<div style='background-color: #e2e8f0; padding: 10px; border-radius: 5px;'>
<h4 style='color: #2c5282;'>Noise Types:</h4>
<ul>
<li><strong>Gaussian</strong>: Adds Gaussian noise (normal distribution) to the image.</li>
<li><strong>Salt and Pepper</strong>: Randomly flips some of the pixel values to 0 or 1.</li>
<li><strong>Speckle</strong>: Adds multiplicative noise to the image.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Noise settings
noise_type = st.sidebar.selectbox("🔊 Noise Type", ["Gaussian", "Salt and Pepper", "Speckle"])
noise_level = st.sidebar.slider("🔈 Noise Level", 0.0, 1.0, 0.5, 0.1, help="Adjust the noise intensity applied to the image")

# Random Prediction button
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

if st.sidebar.button("🎲 Random Prediction", help="Click to select a random image and make predictions"):
    st.session_state.selected_index = np.random.choice(len(x_test))
    st.session_state.x_test_selected = x_test[st.session_state.selected_index:st.session_state.selected_index + 1]
    st.session_state.y_test_selected = y_test[st.session_state.selected_index:st.session_state.selected_index + 1]

# Check if an image has been selected
if st.session_state.selected_index is not None:

    # Generate noisy image
    def add_noise(image, noise_type, noise_level):
        if noise_type == "Gaussian":
            noisy_image = image + np.random.normal(0, noise_level, image.shape)
        elif noise_type == "Salt and Pepper":
            noisy_image = image.copy()
            salt = np.random.random(image.shape) < noise_level / 2
            pepper = np.random.random(image.shape) < noise_level / 2
            noisy_image[salt] = 1
            noisy_image[pepper] = 0
        elif noise_type == "Speckle":
            noisy_image = image + image * np.random.normal(0, noise_level, image.shape)
        return np.clip(noisy_image, 0., 1.)

    x_test_noisy = add_noise(st.session_state.x_test_selected, noise_type, noise_level)

    # Generate predictions
    with st.spinner('🔮 Generating predictions...'):
        autoencoder_pred = autoencoder.predict(x_test_noisy)
        classifier_pred = classifier.predict(x_test_noisy)
        denoise_pred = denoise_and_classify.predict(x_test_noisy)

    # Helper function to plot images
    def plot_images(images, titles):
        fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
        if len(images) == 1:
            axes = [axes]
        for img, ax, title in zip(images, axes, titles):
            ax.imshow(img.reshape(28, 28), cmap='binary')
            ax.set_title(title, fontsize=14)
            ax.axis('off')
        plt.tight_layout()
        return fig

    # Display images
    st.header("📊 Image Comparison")
    fig = plot_images(
        [st.session_state.x_test_selected[0], x_test_noisy[0], autoencoder_pred[0]],
        ["Original", f"Noisy ({noise_type})", "Denoised"]
    )
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)  # Add gap between sections

    # Display classification results
    st.header("🏷️ Classification Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Label", st.session_state.y_test_selected[0], delta=None, help="The actual digit in the image")
    with col2:
        noisy_class = np.argmax(classifier_pred[0])
        st.metric("Noisy Classification", noisy_class, delta=None, help="Predicted digit for the noisy image")
    with col3:
        denoised_class = np.argmax(denoise_pred[0])
        st.metric("Denoised Classification", denoised_class, delta=None, help="Predicted digit after denoising")

    st.markdown("<br>", unsafe_allow_html=True)  # Add gap between sections

    # Create a DataFrame with prediction results
    prediction_data = {
        'True Label': [st.session_state.y_test_selected[0]],
        'Noisy Classification': [noisy_class],
        'Denoised Classification': [denoised_class]
    }
    prediction_df = pd.DataFrame(prediction_data)

    # Display classification probabilities
    st.subheader("📊 Classification Probabilities")
    fig = go.Figure(data=[
        go.Bar(name='Noisy Image', x=list(range(10)), y=classifier_pred[0]),
        go.Bar(name='Denoised Image', x=list(range(10)), y=denoise_pred[0])
    ])
    fig.update_layout(
        barmode='group',
        xaxis_title="Digit",
        yaxis_title="Probability",
        legend_title="Prediction Type",
        title="Classification Probabilities for Noisy vs Denoised Image"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download button for predictions
    st.download_button(
        label="📥 Download Predictions",
        data=prediction_df.to_csv(index=False),
        file_name='mnist_predictions.csv',
        mime='text/csv'
    )

    st.markdown("<br>", unsafe_allow_html=True)  # Add gap between sections

# FAQ section
st.header("FAQ ❓")

with st.expander("What is the MNIST dataset?"):
    st.write(
        "The MNIST dataset contains 60,000 28x28 grayscale images of handwritten digits (0-9) for training and 10,000 images for testing."
    )

with st.expander("What is noise in the context of this app?"):
    st.write(
        "Noise refers to random variations or distortions added to images, which can affect the performance of image classification models."
    )

with st.expander("How does the autoencoder work?"):
    st.write(
        "The autoencoder learns to compress and then reconstruct the input images, which helps in denoising by learning the underlying structure of the data."
    )

with st.expander("Can I use my own images?"):
    st.write(
        "This app currently supports the MNIST dataset only, but you can modify the code to handle other datasets or images."
    )


# Footer
st.markdown("""
<div class='footer'>
    <p style='font-weight: bold;'>🙏 Thank you for using the app!</p>
    <p>For more information, explore the following resources:</p>
    <a href='http://yann.lecun.com/exdb/mnist/' target='_blank'>MNIST Dataset Overview</a>
    <a href='https://www.tensorflow.org/tutorials' target='_blank'>Deep Learning Basics</a>
    <p style='font-size: 18px; margin-top: 10px;'>Developed by <strong>Nagendra Kumar K S</strong> </p>
</div>
""", unsafe_allow_html=True)
