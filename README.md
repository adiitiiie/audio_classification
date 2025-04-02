# Audio Classification using Keras and the UrbanSound8K Dataset

This project demonstrates how to perform audio classification by extracting Mel-Frequency Cepstral Coefficients (MFCC) features from audio files using the UrbanSound8K dataset. A deep neural network built with Keras is used to classify sound clips into one of ten classes, such as "drilling", "siren", or "dog_bark".

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Testing Inference](#testing-inference)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This repository contains a Python script (`audio_classification.py`) that:

- Downloads and extracts the UrbanSound8K dataset.
- Preprocesses audio files by extracting MFCC features using Librosa.
- Builds and trains a deep neural network model using Keras.
- Saves the best model using Keras callbacks.
- Demonstrates how to perform inference on a sample audio file.

The code was originally developed in Google Colab and has been refactored into a standalone, reproducible script.

## Project Structure
The repository is organized as follows:

```
├── README.md                # This file
├── audio_classification.py  # Main Python script for training and inference
├── saved_models/            # Directory where the trained models are saved
└── UrbanSound8K/            # Directory containing the UrbanSound8K dataset (automatically downloaded and extracted)
```

*Note:* The script downloads and extracts the UrbanSound8K dataset if it is not already present in your working directory.

## Prerequisites
Make sure you have the following installed before running the script:

- Python 3.6 or higher
- pip (Python package installer)

### Required Python Libraries:
- TensorFlow (and Keras)
- Librosa
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## Installation

### Clone the repository
```bash
git clone https://github.com/<your_username>/<your_repo_name>.git
cd <your_repo_name>
```

### (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### Install the dependencies
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt` file yet, you can install the main dependencies manually:
```bash
pip install tensorflow keras librosa numpy pandas matplotlib scikit-learn tqdm
```

## Dataset
The project uses the **UrbanSound8K** dataset, which is automatically downloaded within the script using `wget` and extracted using Python's `tarfile` module.

### Notes:
- The dataset is organized into multiple folds (directories). The script iterates through these folders to extract features.
- You can modify the path to the dataset in the script if you wish to use a different location.

## Usage

### Training the Model
Run the main script to start training:
```bash
python audio_classification.py
```
The script will:

- Download and extract the UrbanSound8K dataset (if not already present).
- Read the metadata CSV to locate audio files.
- Extract MFCC features from each audio file.
- Split the dataset into training and testing sets.
- Build a neural network model with multiple Dense layers and dropout.
- Train the model for 100 epochs (default) with a batch size of 32.
- Save the best model checkpoint to the `saved_models/` folder as `audio_classification.keras`.
- Evaluate the model on the test set and print the accuracy.

### Testing Inference
After training, the script also demonstrates inference by:

- Loading a sample audio file (e.g., `UrbanSound8K/drilling_1.wav`).
- Extracting its MFCC features.
- Using the trained model to predict the class.
- Printing the predicted class label to the console.

You can easily adapt the inference logic to test new or custom audio files.

## Model Architecture
The model is a fully connected (Dense) neural network comprising:

- **Input Layer:** Takes a 40-dimensional MFCC feature vector.
- **Hidden Layers:** Three Dense layers with:
  - 100 neurons (first layer), followed by ReLU activation and 50% dropout.
  - 200 neurons (second layer), followed by ReLU activation and 50% dropout.
  - 100 neurons (third layer), followed by ReLU activation and 50% dropout.
- **Output Layer:** Dense layer with 10 neurons and softmax activation (for 10 audio classes).

The model is compiled with the **Adam optimizer** and uses **categorical crossentropy** as the loss function.

## Results
During execution, the script prints the following:

- The shape of the extracted features.
- A summary of data splits (training and testing shapes).
- Detailed model architecture and summary.
- Epoch-wise training progress and validation accuracy.
- Final test accuracy after training.
- Predicted class for the sample audio file during inference.

## Troubleshooting

- **Module not found errors:** Make sure all dependencies are installed using `pip`.
- **Dataset download issues:** Ensure that your environment has internet access if the script needs to download the dataset.
- **GPU/CPU warnings:** TensorFlow might output warnings related to your hardware; these are typically informational.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The **UrbanSound8K** dataset creators.
- The developers of **TensorFlow** and **Keras**.
- The **Librosa** team for their audio analysis library.
- All contributors whose open-source software made this project possible.

