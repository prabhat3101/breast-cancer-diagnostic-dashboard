import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Centered layout prevents ultra-wide stretching during presentations
st.set_page_config(page_title="Breast Cancer Diagnostic Dashboard", layout="centered")

# Load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

benign_sample = [11.5200, 18.7500, 73.3400, 409.0000, 0.0952, 0.0547, 0.0304, 0.0228, 0.1920, 0.0591, 0.3249, 0.9591, 2.1830, 23.4700, 0.0083, 0.0087, 0.0135, 0.0087, 0.0322, 0.0024, 12.8400, 22.4700, 81.8100, 506.2000, 0.1249, 0.0872, 0.0908, 0.0632, 0.3306, 0.0704]
malignant_sample = [12.4500, 15.7000, 82.5700, 477.0000, 0.1278, 0.1700, 0.1578, 0.0809, 0.2087, 0.0761, 0.3345, 0.8902, 2.2170, 27.1900, 0.0075, 0.0335, 0.0367, 0.0114, 0.0217, 0.0051, 15.4700, 23.7500, 103.4000, 741.0000, 0.1791, 0.5249, 0.5355, 0.1741, 0.3985, 0.1244]

st.title("Breast Cancer Diagnostic Dashboard")

tab1, tab2, tab3 = st.tabs(["Patient Diagnosis", "Radar Profile", "Model Insights"])

if 'input_values' not in st.session_state:
    st.session_state.input_values = benign_sample

with tab1:
    st.subheader("Load Quick Sample Data")
    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("Load Benign Sample"):
        st.session_state.input_values = benign_sample
        st.rerun()
    if btn_col2.button("Load Malignant Sample"):
        st.session_state.input_values = malignant_sample
        st.rerun()

    input_data = {}
    
    # Organize 30 inputs into compact collapsible categories
    with st.expander("Mean Features (1-10)", expanded=True):
        cols = st.columns(2)
        for i in range(10):
            feat = features[i]
            val = float(st.session_state.input_values[i])
            input_data[feat] = cols[i % 2].number_input(feat, value=val, format="%.4f")

    with st.expander("Standard Error Features (11-20)", expanded=False):
        cols = st.columns(2)
        for i in range(10, 20):
            feat = features[i]
            val = float(st.session_state.input_values[i])
            input_data[feat] = cols[i % 2].number_input(feat, value=val, format="%.4f")

    with st.expander("Worst Features (21-30)", expanded=False):
        cols = st.columns(2)
        for i in range(20, 30):
            feat = features[i]
            val = float(st.session_state.input_values[i])
            input_data[feat] = cols[i % 2].number_input(feat, value=val, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Diagnostic Prediction", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        malignancy_prob = probabilities[1] * 100

        st.markdown("---")
        if prediction == 1:
            st.error("### Verdict: Malignant")
        else:
            st.success("### Verdict: Benign")

        # Compact Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=malignancy_prob,
            title={'text': "Malignancy Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#d9534f" if prediction == 1 else "#5cb85c"},
                'steps': [
                    {'range': [0, 35], 'color': "#e8f5e9"},
                    {'range': [35, 65], 'color': "#fffde7"},
                    {'range': [65, 100], 'color': "#ffebee"}
                ]
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.subheader("Normalized Cell Radar Comparison")
    st.write("Cell features scaled 0 to 1 for visual comparison.")

    radar_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean']
    
    # Min-Max Scaling Bounds to make all variables visible on spider plot
    min_vals = np.array([6.98, 9.71, 43.79, 0.05, 0.02, 0.00])
    max_vals = np.array([28.11, 39.28, 188.50, 0.16, 0.35, 0.43])
    
    patient_raw = np.array([input_data[f] for f in radar_features])
    benign_raw = np.array([12.14, 17.91, 78.07, 0.087, 0.080, 0.046])
    malignant_raw = np.array([17.46, 21.60, 115.36, 0.102, 0.145, 0.160])

    # Scaling formula
    patient_norm = (patient_raw - min_vals) / (max_vals - min_vals)
    benign_norm = (benign_raw - min_vals) / (max_vals - min_vals)
    malignant_norm = (malignant_raw - min_vals) / (max_vals - min_vals)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=patient_norm, theta=radar_features, fill='toself', name='Current Patient'))
    fig_radar.add_trace(go.Scatterpolar(r=benign_norm, theta=radar_features, name='Average Benign'))
    fig_radar.add_trace(go.Scatterpolar(r=malignant_norm, theta=radar_features, name='Average Malignant'))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.subheader("Feature Importance Rankings")
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    fig_bar = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                     color='Importance', color_continuous_scale='Blues')
    fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig_bar, use_container_width=True)