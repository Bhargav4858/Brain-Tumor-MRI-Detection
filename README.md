# 🧠 Brain Tumor Detection and Classification Using ANN

An educational deep-learning project that classifies brain MRI images into four categories using an Artificial Neural Network (ANN). The project extracts image features with Histogram of Oriented Gradients (HOG), scales the features, and uses a trained ANN model to predict the image class. A Streamlit application provides an interactive interface for uploading MRI images and viewing predictions.

> **Disclaimer:** This project is for educational and research purposes only. It is not a clinically validated medical device and must not be used to diagnose, rule out, or treat a medical condition. MRI images should be interpreted by qualified medical professionals.

---

## 📌 Project Overview

**Input:** Brain MRI image  
**Output:** One of four predicted classes:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

The workflow is:

```text
MRI Image
   │
   ▼
Resize to 64 × 64
   │
   ▼
Grayscale Conversion
   │
   ▼
HOG Feature Extraction
   │
   ▼
StandardScaler
   │
   ▼
Trained ANN Model
   │
   ▼
Predicted Class + Class Scores
```

## 🎯 Objectives

- Explore and understand a brain MRI image dataset.
- Prepare MRI images for machine-learning classification.
- Extract image features using HOG.
- Train an ANN model to classify four MRI image categories.
- Tune model hyperparameters using Optuna.
- Evaluate the trained model on the prepared test data.
- Build an interactive Streamlit application for image prediction.

## 🗂️ Project Structure

Arrange the project files as shown below:

```text
Brain_Tumor_MRI_Detection/
│
├── Dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   │
│   ├── Testing/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   │
│   ├── X_train_trans.csv
│   ├── X_test_trans.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── EDA & Preprocessing.ipynb
├── Model_Building.ipynb
├── model.keras
├── app.py
├── requirements.txt
└── README.md
```

The dataset itself is not included in this repository by default. Add it only if its license and distribution terms permit redistribution. Otherwise, provide instructions for obtaining it from the original source.

## 📊 Dataset

The notebook uses a dataset organized into `Training` and `Testing` folders, with four class subfolders in each.

The EDA notebook records the following class counts:

| Class | Training images | Testing images |
|---|---:|---:|
| Glioma | 1,400 | 400 |
| Meningioma | 1,400 | 400 |
| No Tumor | 1,400 | 400 |
| Pituitary | 1,400 | 400 |
| **Total** | **5,600** | **1,600** |

These counts describe the dataset recorded in the notebook; your local copy may differ.

## ⚙️ Technologies Used

- **Python** – project programming language
- **OpenCV** – image loading, resizing and grayscale conversion
- **NumPy** – numerical operations
- **Pandas** – tabular data handling
- **Matplotlib / Seaborn** – exploratory data visualizations
- **Scikit-image** – HOG feature extraction
- **Scikit-learn** – label encoding, feature scaling and evaluation metrics
- **TensorFlow / Keras** – ANN model development and model saving
- **Optuna** – hyperparameter optimization
- **Streamlit** – interactive web application

## 🧹 Exploratory Data Analysis and Preprocessing

The `EDA & Preprocessing.ipynb` notebook performs the following steps:

1. Inspects the training and testing folders and their classes.
2. Counts images in each class.
3. Displays sample MRI images.
4. Examines image dimensions and class distribution.
5. Resizes images to **64 × 64** pixels.
6. Converts images from BGR to grayscale.
7. Divides grayscale intensity values by **225.0**, as implemented in the notebook.
8. Extracts HOG features with:
   - `orientations = 9`
   - `pixels_per_cell = (8, 8)`
   - `cells_per_block = (2, 2)`
   - `block_norm = "L2-Hys"`
9. Encodes target labels using `LabelEncoder`.
10. Fits `StandardScaler` on training features and transforms the testing features.
11. Exports the transformed features and labels to CSV files.

The resulting HOG feature vector contains **1,764 features per image**.

## 🧠 ANN Model

The `Model_Building.ipynb` notebook builds a feed-forward Artificial Neural Network.

| Layer | Configuration |
|---|---|
| Input | 1,764 features |
| Hidden layer | Dense layer; number of units selected by Optuna |
| Activation | ReLU |
| Regularization | Dropout; rate selected by Optuna |
| Output | 4 neurons |
| Output activation | Softmax |
| Loss function | Sparse categorical cross-entropy |
| Optimizers considered | Adam and SGD |
| Batch size | 128 |
| Maximum training epochs | 100 |

### Hyperparameter Optimization

Optuna is used to search for:

- Hidden-layer units: 64–512
- Dropout rate: 0.1–0.5
- Learning rate: 0.0001–0.01 (log scale)
- Optimizer: Adam or SGD

The objective in the notebook is to maximize validation accuracy.

### Model Evaluation

The notebook evaluates the model using:

- Test loss
- Test accuracy
- Classification report
- Confusion matrix

The trained model is saved as:

```text
model.keras
```

Add your actual evaluation results and screenshots here after running the notebook. No accuracy value is stated in this README because it should be taken from your completed model evaluation.

## ⚠️ Important Notes and Limitations

- This project is an educational prototype, not a medical diagnostic system.
- Model performance depends on the dataset, image quality, preprocessing and training procedure.
- The original notebooks save the trained ANN model but do not save the fitted `StandardScaler`. The Streamlit app therefore refits a scaler from the supplied training images.
- The notebook uses the testing data as validation data during hyperparameter optimization and model fitting, and also evaluates on that data afterward. This means the reported test performance may be optimistic and is not a fully independent final test estimate. A separate validation set and untouched test set are recommended for a rigorous evaluation.
- Confirm that the class-label order used by the application matches the encoder order used during training.
- A model prediction should never replace review by a qualified radiologist or clinician.


## 👤 Author

**Vempa Bhargav**

Add your GitHub profile and contact links here:

- GitHub: `[GitHub](http://github.com/Bhargav4858)`
- LinkedIn: `[Linkedin](https://www.linkedin.com/in/vempa-bhargav-naidu-bab001315/)`

---

⭐ If you find this project useful, consider giving the repository a star.
