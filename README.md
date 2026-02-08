# Bidirectional LSTM for Sequence Classification

## 📌 Project Overview
This project implements a Deep Learning model to classify text sequences using **TensorFlow** and **Keras**. It utilizes a **Bidirectional Long Short-Term Memory (LSTM)** network, which processes sequence data in both forward and backward directions to capture long-term dependencies and context more effectively than standard RNNs.

To ensure optimal performance, this project integrates **Keras Tuner** to automatically search for the best hyperparameters (learning rate, embedding dimension, and LSTM units) rather than relying on manual guesswork.

## 🚀 Key Features
* **Deep Learning Architecture:** Uses an Embedding layer feeding into a Bidirectional LSTM, followed by Dense layers for classification.
* **Automated Optimization:** Implements **Hyperband Tuning** to dynamically select the best model configuration.
* **Robust Training:** Utilizes `EarlyStopping` and `ReduceLROnPlateau` callbacks to prevent overfitting and adapt the learning rate during training.
* **High Performance:** The optimized model achieves **~98.7% accuracy** on the test dataset.

## 🛠️ Technologies Used
* **Python**
* **TensorFlow / Keras** (Deep Learning API)
* **Keras Tuner** (Hyperparameter Optimization)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Data Splitting & Preprocessing)

## 📊 Model Architecture
The model consists of the following layers:
1.  **Embedding Layer:** Converts integer-encoded vocabulary into dense vectors of fixed size.
2.  **Bidirectional LSTM:** Processes the sequence data in both directions to understand context.
3.  **Dropout Layer:** Randomly sets inputs to 0 during training to reduce overfitting.
4.  **Dense Output Layer:** A sigmoid activation function to output a probability (0 to 1).

## 📈 Results
The model was trained with an 80/20 train-test split. 
* **Optimization Metric:** Validation Accuracy
* **Final Test Accuracy:** 98.7%

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BryanStats/LSTM-Predictive-Model.git
   pip install tensorflow pandas scikit-learn keras-tuner
   jupyter notebook "LSTM_Sequence_Classification.ipynb"

