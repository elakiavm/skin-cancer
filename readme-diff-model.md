# Mole Classifier: Skin Cancer Detection using Transfer Learning (`try-with-diff-model.ipynb`)

This project presents a **binary image classification** task to differentiate **benign** and **malignant** skin moles using deep learning. The goal is to build and compare three popular transfer learning models—**VGG16**, **DenseNet201**, and **Xception**—on the ISIC dataset.

---

##  Requirements and Installation

###  Prerequisites
Ensure Python ≥ 3.7 is installed.

### Install Dependencies
Create a virtual environment and install the necessary packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow opencv-python
```

You may optionally use GPU-accelerated TensorFlow for faster training:
```bash
pip install tensorflow-gpu
```

---

##  Dataset Structure

Ensure your dataset directory is structured as follows:

```
dataset/
├── benign/
│   ├── image1.jpg
│   ├── ...
├── malignant/
│   ├── image1.jpg
│   ├── ...
```

Images should be resized to **224x224** RGB format before training.

---

## Project Structure & Notebook Explanation

| Section | Description |
|---------|-------------|
| **Step 1: Import Libraries** | Loads essential libraries for image processing, data handling, visualization, modeling (TensorFlow/Keras), and evaluation. |
| **Step 2: Load and Preprocess Data** | Loads images and labels from disk. Images are resized and converted to arrays. Pixel values are normalized using dataset mean and standard deviation. |
| **Step 3: Data Augmentation** | Implements real-time data augmentation using `ImageDataGenerator` to prevent overfitting. |
| **Step 4: Build Transfer Learning Models** | Builds CNN classifiers using pretrained backbones (VGG16, DenseNet201, Xception) with top layers for binary classification. Models are frozen initially. |
| **Step 5: Training Helpers** | Includes helper functions to plot learning curves and confusion matrices. |
| **Step 6: Train and Evaluate** | Each model is trained using the augmented data. Performance metrics like accuracy, confusion matrix, and classification report are displayed. |
| **Step 7: Model Comparison** | Plots a graph comparing validation accuracy across all models for easy interpretation. |

---

## Results Summary

| Model       | Final Validation Accuracy | Observations |
|-------------|---------------------------|--------------|
| **VGG16**     | ~0.75                        | High early accuracy but slight overfitting by later epochs. |
| **DenseNet201** | **~0.78 (Best)**               | Smooth learning curve and stable validation performance. |
| **Xception**   | ~0.76                        | Balanced performance and no significant overfitting. |

---

## Conclusion

- **DenseNet201** outperformed other models in validation accuracy and stability.
- **VGG16** showed early overfitting, requiring either fine-tuning or more regularization.
- **Xception** delivered consistent results and can be considered a strong secondary candidate.

