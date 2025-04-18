
# Skin Cancer Classification using Basic CNN (`basic.ipynb`)

This project implements a **basic Convolutional Neural Network (CNN)** to classify dermoscopic skin mole images into two categories:
- **Benign** (non-cancerous)
- **Malignant** (cancerous)

The CNN model serves as a foundational baseline to understand skin cancer image classification.

---

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── benign/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── malignant/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Each subfolder contains respective images of skin moles classified by medical professionals.

---

## Requirements

Make sure the following Python libraries are installed:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow opencv-python pillow
```

---

## Notebook Breakdown: `basic.ipynb`

| Step | Description |
|------|-------------|
| **1. Import Libraries** | Loads essential libraries for deep learning, image processing, and visualization. |
| **2. Image Enhancement (CLAHE)** | Applies **Contrast Limited Adaptive Histogram Equalization** using OpenCV to enhance image contrast. |
| **3. Data Loading** | Loads images from `benign` and `malignant` folders, resizes them to **224×224**, converts to RGB, applies CLAHE, and saves them as arrays. |
| **4. Label Encoding** | Encodes labels as integers and converts them to one-hot encoding using `to_categorical`. |
| **5. Data Normalization** | Scales the image data using standardization (zero mean and unit variance). |
| **6. Data Augmentation** | Uses `ImageDataGenerator` to apply transformations such as rotation, flip, zoom, etc., to improve generalization. |
| **7. Model Architecture (Basic CNN)** | Builds a custom CNN model with:
  - Conv2D layers
  - MaxPooling
  - Flatten + Dense
  - Dropout for regularization |
| **8. Model Training** | Trains the CNN for a few epochs, tracking training and validation performance. |
| **9. Evaluation** | Evaluates the model using:
  - Accuracy score
  - Confusion matrix
  - Classification report
  - Learning curves |

---

## Results & Visualization

The model performance is visualized using:
- **Training & Validation Accuracy and Loss curves**
- **Confusion Matrix**
- **Classification Report** with Precision, Recall, and F1-score

These tools help assess model reliability and diagnose overfitting or underfitting.

---

## Observations

- **CLAHE preprocessing** significantly boosts visibility of image features.
- Model shows good early convergence but may need deeper layers or more epochs for better generalization.
- Useful as a **baseline model** for further improvements using Transfer Learning.

