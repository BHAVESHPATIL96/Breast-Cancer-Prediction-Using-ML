import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('brest_cancer.pkl', 'rb'))

# Title
st.title('🔬 Breast Cancer Prediction App')
st.write("This app predicts whether a tumor is **Malignant** or **Benign** based on cell nuclei features.")

# Sidebar input
st.sidebar.header('🧬 Input Tumor Characteristics')

def user_input_features():
    mean_radius = st.sidebar.number_input('Mean Radius', 6.0, 30.0, 14.0)
    mean_texture = st.sidebar.number_input('Mean Texture', 9.0, 40.0, 20.0)
    mean_perimeter = st.sidebar.number_input('Mean Perimeter', 40.0, 190.0, 90.0)
    mean_area = st.sidebar.number_input('Mean Area', 150.0, 2500.0, 500.0)
    mean_smoothness = st.sidebar.number_input('Mean Smoothness', 0.05, 0.16, 0.1)
    mean_compactness = st.sidebar.number_input('Mean Compactness', 0.02, 0.35, 0.1)
    mean_concavity = st.sidebar.number_input('Mean Concavity', 0.0, 0.45, 0.1)
    mean_concave_points = st.sidebar.number_input('Mean Concave Points', 0.0, 0.2, 0.05)
    mean_symmetry = st.sidebar.number_input('Mean Symmetry', 0.1, 0.3, 0.2)
    mean_fractal_dimension = st.sidebar.number_input('Mean Fractal Dimension', 0.04, 0.1, 0.06)
    
    radius_error = st.sidebar.number_input('Radius Error', 0.1, 3.0, 1.0)
    texture_error = st.sidebar.number_input('Texture Error', 0.3, 5.0, 1.5)
    perimeter_error = st.sidebar.number_input('Perimeter Error', 1.0, 30.0, 5.0)
    area_error = st.sidebar.number_input('Area Error', 6.0, 550.0, 40.0)
    smoothness_error = st.sidebar.number_input('Smoothness Error', 0.002, 0.03, 0.01)
    compactness_error = st.sidebar.number_input('Compactness Error', 0.002, 0.15, 0.02)
    concavity_error = st.sidebar.number_input('Concavity Error', 0.0, 0.4, 0.02)
    concave_points_error = st.sidebar.number_input('Concave Points Error', 0.0, 0.05, 0.01)
    symmetry_error = st.sidebar.number_input('Symmetry Error', 0.007, 0.08, 0.02)
    fractal_dimension_error = st.sidebar.number_input('Fractal Dimension Error', 0.001, 0.03, 0.01)
    
    worst_radius = st.sidebar.number_input('Worst Radius', 7.0, 40.0, 16.0)
    worst_texture = st.sidebar.number_input('Worst Texture', 12.0, 50.0, 25.0)
    
    data = np.array([
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
        radius_error, texture_error, perimeter_error, area_error, smoothness_error,
        compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture
    ]).reshape(1, -1)
    
    return data

input_features = user_input_features()

# Predict
if st.button('Predict'):
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    if prediction[0] == 1:
        st.error(f'🚨 Prediction: Malignant Tumor\n\n🔵 Confidence: {round(prediction_proba[0][1]*100,2)}%')
    else:
        st.success(f'✅ Prediction: Benign Tumor\n\n🟢 Confidence: {round(prediction_proba[0][0]*100,2)}%')

st.markdown("---")
st.caption("Built with ❤️ ")

