import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from skimage.feature import hog


# =========================
# Paths and Constants
# =========================

IMG_SIZE_ML = 128

MODEL_PATH = os.path.join("saved_models", "svm_hog.pkl")
SCALER_PATH = os.path.join("saved_models", "hog_scaler.pkl")
ENCODER_PATH = os.path.join("saved_models", "label_encoder.pkl")

RESULTS_PATH = os.path.join("results", "final_models_comparison.csv")
REPORT_PATH = os.path.join("reports", "svm_report.txt")


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Eye Disease Classifier",
    layout="wide"
)


# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top, #111827 0%, #050816 42%, #020617 100%);
        color: #E5E7EB;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .main-title span {
        color: #38BDF8;
    }

    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .nav-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 35px;
    }

    .hero-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 28px;
        padding: 36px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 30px;
    }

    .upload-box {
        border: 2px dashed rgba(56, 189, 248, 0.35);
        border-radius: 24px;
        padding: 35px;
        background: rgba(2, 6, 23, 0.45);
        text-align: center;
        margin-bottom: 20px;
    }

    .section-title {
        text-align: center;
        font-size: 30px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 35px;
        margin-bottom: 22px;
    }

    .info-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 24px;
        min-height: 185px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .info-card h3 {
        color: #38BDF8;
        font-size: 18px;
        margin-bottom: 10px;
    }

    .info-card p {
        color: #CBD5E1;
        line-height: 1.7;
        font-size: 14px;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(37, 99, 235, 0.12));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 22px;
        padding: 22px;
        text-align: center;
    }

    .metric-card h2 {
        color: #F8FAFC;
        font-size: 32px;
        margin: 0;
    }

    .metric-card p {
        color: #93C5FD;
        margin: 6px 0 0 0;
        font-size: 14px;
    }

    .prediction-card {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 24px;
        padding: 26px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
    }

    .prediction-label {
        font-size: 34px;
        font-weight: 800;
        color: #22C55E;
        margin-bottom: 8px;
    }

    .small-muted {
        color: #94A3B8;
        font-size: 13px;
    }

    .warning-box {
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.35);
        color: #FDE68A;
        padding: 16px 20px;
        border-radius: 16px;
        margin-top: 18px;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        margin-top: 80px;
        padding-bottom: 20px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 14px;
        padding: 10px 22px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(37, 99, 235, 0.35);
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.75);
        border-radius: 18px;
        padding: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Load Artifacts
# =========================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


def format_label(label):
    return label.replace("_", " ").title()


def extract_hog_features(uploaded_file, img_size=128):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    image_resized = cv2.resize(image_np, (img_size, img_size))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1), image


def decision_scores_to_confidence(scores):
    scores = np.array(scores, dtype=float)
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / exp_scores.sum()
    return probabilities


def show_footer():
    st.markdown(
        """
        <div class="footer">
            Designed and built for Eye Disease Classification Project<br>
            Created by Maybe! Team - 2026
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Check Required Files + Init App
# =========================

missing_files = []

for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
    if not os.path.exists(path):
        missing_files.append(path)


def init_app():
    if missing_files:
        st.error("Some required model files are missing:")
        for file in missing_files:
            st.write(file)
        st.stop()

    return load_artifacts()


model, scaler, label_encoder = init_app()



# =========================
# Header
# =========================

st.markdown(
    """
    <div class="main-title">
        Eye Disease <span>Classification</span>
    </div>
    <div class="subtitle">
        Retinal image analysis using Machine Learning and Deep Learning models
    </div>
    """,
    unsafe_allow_html=True
)


tabs = st.tabs(["Home", "Classifier", "Insights", "About"])


# =========================
# Home Page
# =========================

with tabs[0]:
    st.markdown(
        """
        <div class="hero-card">
            <h2 style="text-align:center; color:#F8FAFC; margin-bottom:10px;">
                AI-Based Retinal Disease Screening
            </h2>
            <p style="text-align:center; color:#CBD5E1; font-size:16px; line-height:1.7;">
                This project classifies retinal fundus images into four categories:
                Cataract, Diabetic Retinopathy, Glaucoma, and Normal.
                The final deployed model is based on HOG feature extraction and SVM classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Dataset</h3>
                <p>
                The dataset contains retinal images distributed across four classes:
                Cataract, Diabetic Retinopathy, Glaucoma, and Normal.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Modeling Approach</h3>
                <p>
                Six models were compared: three traditional machine learning models
                and three deep learning models.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Selected Model</h3>
                <p>
                SVM + HOG was selected for deployment because it achieved the best
                test accuracy and is lightweight for CPU deployment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Best Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            <div class="metric-card">
                <h2>85.71%</h2>
                <p>Test Accuracy</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <h2>0.85</h2>
                <p>Weighted F1-Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            """
            <div class="metric-card">
                <h2>SVM + HOG</h2>
                <p>Deployment Model</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    show_footer()


# =========================
# Classifier Page
# =========================

with tabs[1]:
    st.markdown('<div class="section-title">Retinal Image Classifier</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-card">
            <div class="upload-box">
                <h3 style="color:#F8FAFC;">Upload Retinal Image</h3>
                <p style="color:#94A3B8;">Drag and drop or browse an image file</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        features, original_image = extract_hog_features(uploaded_file, IMG_SIZE_ML)
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(original_image, caption="Uploaded Image", use_container_width=True)

        with col_result:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="small-muted">Prediction Result</div>
                    <div class="prediction-label">{format_label(predicted_class)}</div>
                    <p style="color:#CBD5E1;">
                        The uploaded retinal image was classified using the selected
                        SVM + HOG model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="warning-box">
                    This result is for educational purposes only and must not be used as a medical diagnosis.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">Model Scores</div>', unsafe_allow_html=True)

        if hasattr(model, "decision_function"):
            scores = model.decision_function(scaled_features)[0]
            confidence = decision_scores_to_confidence(scores)

            score_df = pd.DataFrame({
                "Class": [format_label(c) for c in label_encoder.classes_],
                "Decision Score": scores,
                "Relative Confidence": confidence
            }).sort_values(by="Relative Confidence", ascending=False)

            st.dataframe(score_df, use_container_width=True)

            chart_df = score_df.set_index("Class")[["Relative Confidence"]]
            st.bar_chart(chart_df)

        st.markdown(
            """
            <div class="hero-card">
                <h3 style="color:#F8FAFC;">Prediction Pipeline</h3>
                <p style="color:#CBD5E1; line-height:1.8;">
                Image Upload → Resize to 128×128 → Grayscale Conversion →
                HOG Feature Extraction → Feature Scaling → SVM Prediction
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    show_footer()


# =========================
# Insights Page
# =========================

with tabs[2]:
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-card">
            <h3 style="color:#F8FAFC;">Training Performance and Model Comparison</h3>
            <p style="color:#CBD5E1;">
            The project compared six models and selected the best model based on
            test accuracy, weighted F1-score, glaucoma recall, and deployment suitability.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.path.exists(RESULTS_PATH):
        results_df = pd.read_csv(RESULTS_PATH)

        st.subheader("Final Models Comparison")
        st.dataframe(results_df, use_container_width=True)

        if "Test Accuracy" in results_df.columns:
            chart_df = results_df[["Model", "Test Accuracy"]].set_index("Model")
            st.bar_chart(chart_df)

    else:
        st.warning("Final models comparison file was not found.")

    st.subheader("Selected Model Report: SVM + HOG")

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            report = f.read()
        st.code(report)
    else:
        st.warning("SVM classification report was not found.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Why SVM + HOG?</h3>
                <p>
                It achieved the highest test accuracy among all six models and
                requires less computational power compared to deep learning models.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Medical Observation</h3>
                <p>
                Glaucoma was the most challenging class across several models,
                often being confused with the Normal class.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    show_footer()


# =========================
# About Page
# =========================

with tabs[3]:
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-card">
            <h3 style="color:#F8FAFC;">Project Overview</h3>
            <p style="color:#CBD5E1; line-height:1.8;">
            This project aims to classify retinal fundus images into four categories:
            Cataract, Diabetic Retinopathy, Glaucoma, and Normal.
            The system compares traditional machine learning and deep learning models,
            then deploys the best-performing and most practical model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Machine Learning Models</h3>
                <p>
                Logistic Regression, SVM, and Random Forest were trained using
                HOG features extracted from retinal images.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Deep Learning Models</h3>
                <p>
                Lightweight CNN, MobileNetV2, and EfficientNetB0 were trained
                and evaluated as deep learning approaches.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Deployment</h3>
                <p>
                The final application is built using Streamlit and can be
                containerized using Docker for reliable deployment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="warning-box">
            Disclaimer: This application is an academic project and should not be used
            as a substitute for professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )

    show_footer()