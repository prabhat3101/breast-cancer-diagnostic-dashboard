# Breast Cancer Diagnostic Dashboard 🩺

An interactive Machine Learning web application built using Python, Streamlit, Scikit-Learn, and Plotly to predict breast cancer diagnosis (Benign vs. Malignant) based on 30 clinical cell nuclear features from fine-needle aspirate (FNA) images.

---

## Key Features

* **Interactive Diagnostic Prediction:** Accepts 30 clinical parameters organized across expandable feature drawers to output real-time verdicts with a custom Plotly malignancy risk gauge (0–100%).
* **Normalized Cell Radar Comparison:** Visualizes min-max scaled patient metrics overlaid against average Benign and Malignant baseline footprints.
* **Model Insights & Explainability:** Displays global feature importance rankings driven by a trained Random Forest classifier.
* **Quick-Load Sample Data:** Preset test buttons for instant testing of Benign and Malignant patient profiles.

---

## Tech Stack

* **Frontend / UI:** Streamlit, Plotly (Interactive Gauges & Radar Charts)
* **Machine Learning:** Scikit-Learn (Random Forest, StandardScaler, LabelEncoder)
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Joblib

---

## Repository Structure

```text
├── app.py               # Main Streamlit dashboard application
├── train_model.py       # Script for preprocessing, training, and saving model
├── cancer_model.pkl     # Pre-trained RandomForestClassifier model
├── scaler.pkl           # Pre-fitted StandardScaler instance
├── data.csv             # Wisconsin Breast Cancer Dataset
└── README.md            # Project documentation
