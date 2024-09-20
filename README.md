# MNIST Denoising and Classification with Autoencoders

This project demonstrates the use of autoencoders for image denoising and classification improvement on the MNIST dataset. It includes a Jupyter notebook for model training and a Streamlit web application for interactive visualization of the results.

## 🌟 Features

- Image denoising using autoencoders
- MNIST digit classification
- Interactive web application for visualization
- Support for multiple noise types (Gaussian, Salt and Pepper, Speckle)
- Adjustable noise levels
- Comparison of classification results before and after denoising

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- Streamlit
- Matplotlib
- Plotly
- Pandas
- NumPy

## 📁 Project Structure

- `ImageNoiseReductionAutoencoders.ipynb`: Jupyter notebook containing the model training code
- `app.py`: Streamlit application for interactive visualization
- `autoencoder_model.h5`: Trained autoencoder model
- `classifier_model.h5`: Trained classifier model
- `denoise_and_classify.h5`: Combined model for denoising and classification
- `requirements.txt`: List of Python dependencies

## 🚀 Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/your-username/mnist-denoising-classification.git
   cd mnist-denoising-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## 📘 How to Use the App

1. Use the sidebar to select the noise type and adjust the noise level.
2. Click the "Random Prediction" button to select a random image from the MNIST test set.
3. View the original, noisy, and denoised images.
4. Compare the classification results for the noisy and denoised images.
5. Explore the classification probabilities graph for more detailed insights.
6. Download the prediction results as a CSV file if desired.

## 🧠 Model Architecture

- **Autoencoder**: A simple autoencoder with one hidden layer (64 units) for denoising.
- **Classifier**: A feedforward neural network with two hidden layers (256 units each) for digit classification.
- **Combined Model**: The autoencoder and classifier are combined to create a pipeline for denoising and classification.

## 📊 Results

The project demonstrates improved classification accuracy on noisy MNIST images after applying the autoencoder for denoising. The web application allows for interactive exploration of these results.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/your-username/mnist-denoising-classification/issues) if you want to contribute.



## 👤 Author

**Nagendra Kumar K S**

- GitHub: [(https://github.com/Nagendra2k00)]
- LinkedIn: (https://linkedin.com/in/nagendrakumarks)

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

Don't forget to star ⭐ this repository if you found it helpful!
