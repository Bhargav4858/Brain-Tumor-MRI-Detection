
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os

from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# 2. PROJECT CONFIGURATION
# ==========================================

MODEL_PATH = "model.keras"
TRAIN_PATH = os.path.join("Dataset", "Training")

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMG_SIZE = 64

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #4A90E2;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: gray;
    font-size: 17px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: rgba(128,128,128,0.10);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ==========================================
# 5. IMAGE FEATURE EXTRACTION
# ==========================================

def extract_features(image):

    # Resize image
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert BGR to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize image
    gray = gray / 225.0

    # HOG feature extraction
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ==========================================
# 6. FIT SCALER USING TRAINING IMAGES
# ==========================================

@st.cache_resource
def load_scaler():

    X_train = []

    for folder in CLASS_NAMES:

        # Folder names are lowercase
        folder_path = os.path.join(
            TRAIN_PATH,
            folder.lower().replace(" ", "")
        )

        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                f"Training folder not found: {folder_path}"
            )

        for img_name in os.listdir(folder_path):

            img_path = os.path.join(
                folder_path,
                img_name
            )

            img = cv2.imread(img_path)

            if img is None:
                continue

            features = extract_features(img)

            X_train.append(features)

    X_train = np.array(X_train)

    if len(X_train) == 0:
        raise ValueError("No training images found.")

    if X_train.shape[1] != 1764:
        raise ValueError(
            f"Expected 1764 features, got {X_train.shape[1]}"
        )

    scaler = StandardScaler()

    scaler.fit(X_train)

    return scaler


# ==========================================
# 7. PREDICTION FUNCTION
# ==========================================

def predict_tumor(image, model, scaler):

    # Extract HOG features
    features = extract_features(image)

    # Reshape
    features = features.reshape(1, -1)

    # Feature scaling
    features = scaler.transform(features)

    # Model prediction
    prediction = model.predict(
        features,
        verbose=0
    )

    # Predicted class
    predicted_class = np.argmax(
        prediction,
        axis=1
    )[0]

    # Confidence scores
    probabilities = prediction[0]

    return predicted_class, probabilities


# ==========================================
# 8. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🧠 MRI Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Tumor Prediction",
        "About Model",
        "Prediction History"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Brain Tumor Detection using ANN and HOG"
)


# ==========================================
# 9. LOAD MODEL AND SCALER
# ==========================================

@st.cache_resource
def load_assets():

    model = load_model()
    scaler = load_scaler()

    return model, scaler


# ==========================================
# 10. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Brain Tumor MRI Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">AI-powered MRI Image Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tumor Categories",
        "3"
    )

    col2.metric(
        "Total Classes",
        "4"
    )

    col3.metric(
        "Image Size",
        "64 × 64"
    )

    st.markdown("## Project Overview")

    st.write("""
    This application uses an Artificial Neural Network
    to classify brain MRI images into four categories.

    The application performs the following operations:

    1. Upload MRI image.
    2. Resize image to 64 × 64.
    3. Convert image into grayscale.
    4. Extract HOG features.
    5. Apply StandardScaler.
    6. Predict tumor category using ANN.
    """)

    st.markdown("## Supported Categories")

    c1, c2, c3, c4 = st.columns(4)

    c1.info("Glioma")
    c2.info("Meningioma")
    c3.success("No Tumor")
    c4.info("Pituitary")


# ==========================================
# 11. TUMOR PREDICTION PAGE
# ==========================================

elif page == "Tumor Prediction":

    st.title("🔍 Brain Tumor Prediction")

    st.write(
        "Upload a brain MRI image to predict its category."
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Read uploaded image
        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_rgb = np.array(pil_image)

        # Display image
        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Uploaded MRI Image")

            st.image(
                image_rgb,
                use_container_width=True
            )

        with col2:

            st.subheader("Prediction")

            if st.button(
                "Predict Tumor",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Analyzing MRI image..."
                    ):

                        model, scaler = load_assets()

                        # Convert RGB to BGR
                        image_bgr = cv2.cvtColor(
                            image_rgb,
                            cv2.COLOR_RGB2BGR
                        )

                        predicted_class, probabilities = (
                            predict_tumor(
                                image_bgr,
                                model,
                                scaler
                            )
                        )

                    # Get class name
                    result = CLASS_NAMES[
                        predicted_class
                    ]

                    confidence = (
                        probabilities[predicted_class] * 100
                    )

                    # Display prediction
                    st.markdown(
                        '<div class="result-card">',
                        unsafe_allow_html=True
                    )

                    st.success(
                        f"Prediction: {result}"
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    # Probability chart
                    st.subheader(
                        "Class Probability Analysis"
                    )

                    probability_df = pd.DataFrame({
                        "Class": CLASS_NAMES,
                        "Probability": probabilities
                    })

                    st.bar_chart(
                        probability_df.set_index("Class")
                    )

                    # Save history
                    if "history" not in st.session_state:
                        st.session_state.history = []

                    st.session_state.history.append({
                        "Image": uploaded_file.name,
                        "Prediction": result,
                        "Confidence": round(
                            float(confidence), 2
                        )
                    })

                except Exception as e:

                    st.error(
                        f"Prediction Error: {e}"
                    )


# ==========================================
# 12. ABOUT MODEL PAGE
# ==========================================

elif page == "About Model":

    st.title("About the ANN Model")

    st.write("""
    The model uses an Artificial Neural Network
    for brain tumor classification.
    """)

    st.subheader("Model Architecture")

    st.code("""
    Input Layer       : 1764 Features

    Hidden Layer      : 468 Neurons

    Activation        : ReLU

    Dropout           : Hyperparameter tuned

    Output Layer      : 4 Neurons

    Activation        : Softmax

    Optimizer         : Adam / SGD

    Loss Function     : Sparse Categorical Crossentropy
    """)

    st.subheader("Feature Extraction")

    st.write("""
    HOG (Histogram of Oriented Gradients)
    is used to extract image shape and gradient
    information from MRI images.
    """)


# ==========================================
# 13. PREDICTION HISTORY PAGE
# ==========================================

elif page == "Prediction History":

    st.title("Prediction History")

    if (
        "history" in st.session_state
        and len(st.session_state.history) > 0
    ):

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # Download history
        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download History CSV",
            csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available."
        )


# ==========================================
# 14. FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Brain Tumor MRI Detection | Deep Learning Project"
)

st.warning(
    "Educational demonstration only. This model is not "
    "clinically validated and must not be used for medical "
    "diagnosis or treatment decisions. Consult a qualified "
    "medical professional for MRI interpretation."
)
