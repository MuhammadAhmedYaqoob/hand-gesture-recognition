# Hand Gesture Recognition System

A complete system for recognizing hand gestures from images using MediaPipe and deep learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the System](#running-the-system)
  - [Verifying Setup](#verifying-setup)
  - [Training](#training)
  - [Testing](#testing)
  - [Real-time Detection](#real-time-detection)
- [Dataset Organization](#dataset-organization)
- [Model Output](#model-output)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a hand gesture recognition system that can identify different hand gestures from images. The system detects hand landmarks using MediaPipe and classifies them using a deep learning model. It includes tools for data augmentation, training, and real-time inference.

## Features

- Hand detection and landmark extraction
- Advanced data augmentation to expand small datasets
- Model training with either MobileNet or ResNet
- Real-time gesture recognition via webcam
- Visualization of results and performance metrics

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended but not required)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
hand-gesture-recognition/
├── gesture_recognition_pipeline.py  # Training pipeline
├── gesture_recognition_inference.py # Inference module
├── setup_verification.py           # Dataset verification tool
├── requirements.txt                # Dependencies
├── model/                          # Saved models and results (created during training)
└── README.md                       # This file
```

## Running the System

### Verifying Setup

Before training, verify your dataset and MediaPipe setup:

```bash
python setup_verification.py
```

This will check your folder structure and test MediaPipe hand detection on sample images.

### Training

To train the model:

```bash
python gesture_recognition_pipeline.py
```

The training process will:
1. Load your dataset
2. Apply data augmentation
3. Train a gesture recognition model
4. Save the model and evaluation results to the `model` directory

### Testing

To test your trained model on new images:

```bash
python gesture_recognition_inference.py
```

By default, this will:
- Load your trained model from the `model` directory
- Process a sample image
- Display the prediction

You can modify the script to test on your own images by changing the `image_path` variable.

### Real-time Detection

For real-time detection using your webcam, uncomment the following line in `gesture_recognition_inference.py`:

```python
# recognizer.real_time_detection()
```

Then run:

```bash
python gesture_recognition_inference.py
```

The webcam will open, and the system will display gesture predictions in real-time. Press 'q' to quit.

## Dataset Organization

The dataset should be organized in the following structure:
```
dataset_root/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
├── valid/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

You need to update the `BASE_PATH` variable in the scripts to point to your dataset location.

## Model Output

After training, the system will generate:

1. `final_model.h5` - The trained model
2. `label_encoder.pkl` - The class labels mapping
3. `confusion_matrix.png` - Visualization of model performance
4. `training_history.png` - Training progress graph
5. `inference_samples/` - Directory with example predictions

All outputs are saved to the `model` directory.

## Troubleshooting

1. **Enviroment**:
   - You are advised to use virtual enviroment.

2. **Dependency Issues**:
   - Use python 3.10.7 or any compatible version with requirment.txt