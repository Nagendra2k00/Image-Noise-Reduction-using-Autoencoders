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
- `requirements.txt`: List of Python dependencies

## 🚀 Getting Started

1. Clone the repository:
   ```
   git clone [https://github.com/Nagendra2k00/Image-Noise-Reduction-using-Autoencoders.git]
   cd mnist-denoising-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   - Open `ImageNoiseReductionAutoencoders.ipynb` in Jupyter Lab or Jupyter Notebook.
   - Run all cells in the notebook. This will train the models and save them as `.h5` files:
     - `autoencoder_model.h5`
     - `classifier_model.h5`
     - `denoise_and_classify.h5`

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## 📘 How to Use the App

1. Use the sidebar to select the noise type and adjust the noise level.
2. Click the "Random Prediction" button to select a random image from the MNIST test set.
3. View the original, noisy, and denoised images.
4. Compare the classification results for the noisy and denoised images.
5. Explore the classification probabilities graph for more detailed insights.
6. Download the prediction results as a CSV file if desired.

## 🖼️ Screenshots

### Main Application Interface
![image](https://github.com/user-attachments/assets/c129c6af-623d-40fe-933e-2b21031bcab6)

### Noise Types
![image](https://github.com/user-attachments/assets/6bfcacd6-844b-40d1-8d2f-83ab30982339)

### Classification Results
![image](https://github.com/user-attachments/assets/845706d1-ba11-4b0e-bb53-f57eb973b564)

![image](https://github.com/user-attachments/assets/6d6c6b78-6e52-42a3-a47b-388d2015e631)

### Probability Distribution Graph
![plot](https://github.com/user-attachments/assets/80be9852-ec3a-4df7-a1b2-fd0c8c5dcf20)



## 🧠 Model Architecture

- **Autoencoder**: A simple autoencoder with one hidden layer (64 units) for denoising.
- **Classifier**: A feedforward neural network with two hidden layers (256 units each) for digit classification.
- **Combined Model**: The autoencoder and classifier are combined to create a pipeline for denoising and classification.

## 📊 Results

The project demonstrates improved classification accuracy on noisy MNIST images after applying the autoencoder for denoising. The web application allows for interactive exploration of these results.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! if you want to contribute.


## 👤 Author

**Nagendra Kumar K S**

- GitHub: [@Nagendra Kumar K S](https://github.com/Nagendra2k00)
- LinkedIn: [@Nagendra Kumar K S](https://linkedin.com/in/nagendrakumarks)

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

Don't forget to star ⭐ this repository if you found it helpful!
