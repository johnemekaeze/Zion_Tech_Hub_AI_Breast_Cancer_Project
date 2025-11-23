
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Zion Tech Hub - AI Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### Zion Tech Hub AI Breast Cancer Detection System\nPowered by Advanced Machine Learning"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main background and text */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        min-height: 100vh;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f7fa 100%);
        border-right: 2px solid #3370E7;
    }
    
    [data-testid="stSidebarContent"] {
        color: #1a1a1a;
    }
    
    /* Title styling */
    h1 {
        color: #3370E7;
        text-shadow: 1px 1px 2px rgba(51, 112, 231, 0.1);
        font-size: 2.5em !important;
        margin-bottom: 10px !important;
    }
    
    h2, h3 {
        color: #3370E7;
        text-shadow: 1px 1px 2px rgba(51, 112, 231, 0.08);
    }
    
    /* Subtitle */
    .subtitle {
        color: #555555;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    /* Input containers */
    [data-testid="stNumberInput"] > div > div > input {
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stNumberInput"] > div > div > input:hover {
        border-color: #3370E7 !important;
        box-shadow: 0 0 8px rgba(51, 112, 231, 0.2) !important;
    }
    
    [data-testid="stNumberInput"] > div > div > input:focus {
        border-color: #3370E7 !important;
        box-shadow: 0 0 12px rgba(51, 112, 231, 0.4) !important;
    }
    
    /* Button styling - Primary buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3370E7 0%, #2563d8 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 14px 32px !important;
        font-size: 1.05em !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(51, 112, 231, 0.3) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(51, 112, 231, 0.5) !important;
        background: linear-gradient(135deg, #2563d8 0%, #1e4fb8 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(51, 112, 231, 0.4) !important;
    }
    
    /* Success and error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.05)) !important;
        color: #2e7d32 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        border-left: 5px solid #4CAF50 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(244, 67, 54, 0.05)) !important;
        color: #c62828 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        border: 2px solid #f44336 !important;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ffffff, #f9fafb);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(51, 112, 231, 0.1);
        border-left: 4px solid #3370E7;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f9fafb);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(51, 112, 231, 0.1);
        margin: 10px 0;
        border: 2px solid #3370E7;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #3370E7;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar labels */
    .sidebar-title {
        color: #3370E7 !important;
        font-size: 1.3em !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
    }
    
    /* Data table styling */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #3370E7 !important;
        opacity: 0.3;
        margin: 30px 0 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6 !important;
        color: #666 !important;
        border-radius: 8px 8px 0 0 !important;
        border-bottom: 3px solid transparent !important;
        transition: all 0.3s ease !important;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8ecf1 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3370E7 !important;
        color: white !important;
        border-bottom: 3px solid #3370E7 !important;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] > div:first-child > button {
        background-color: #f0f2f6 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stExpander"] > div:first-child > button:hover {
        background-color: #e8ecf1 !important;
        border-left: 4px solid #3370E7 !important;
    }
    
    /* Clear button styling */
    .clear-btn {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-size: 0.95em !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.2) !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }
    
    .clear-btn:hover {
        background: linear-gradient(135deg, #ff5252 0%, #ff4444 100%) !important;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Benign result styling */
    .benign-result {
        background: linear-gradient(135deg, #51cf66, #37b24d);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);
        border: 2px solid #37b24d;
    }
    
    /* Malignant result styling */
    .malignant-result {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        border: 2px solid #ff5252;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('best_mlp.keras')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# Feature names (from the original dataset)
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #3370E7;'>🔬 Zion Tech Hub AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #3370E7;'>Breast Cancer Detection System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555555; font-size: 1.05em;'>Advanced ML-Powered Diagnostic Tool</p>", unsafe_allow_html=True)

st.markdown("---")

# Default values for testing
default_values = {
    'mean radius': 12.5, 'mean texture': 18.3, 'mean perimeter': 78.9, 'mean area': 490.5, 'mean smoothness': 0.098,
    'mean compactness': 0.085, 'mean concavity': 0.045, 'mean concave points': 0.032, 'mean symmetry': 0.175,
    'mean fractal dimension': 0.062, 'radius error': 0.35, 'texture error': 0.82, 'perimeter error': 2.45,
    'area error': 28.5, 'smoothness error': 0.0065, 'compactness error': 0.015, 'concavity error': 0.018,
    'concave points error': 0.0055, 'symmetry error': 0.0125, 'fractal dimension error': 0.0035,
    'worst radius': 13.8, 'worst texture': 22.5, 'worst perimeter': 85.2, 'worst area': 560.8, 'worst smoothness': 0.128,
    'worst compactness': 0.115, 'worst concavity': 0.065, 'worst concave points': 0.045, 'worst symmetry': 0.210,
    'worst fractal dimension': 0.075
}

# Main content in tabs
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Information", "❓ About"])

with tab1:
    st.markdown("<p style='color: #3370E7; font-size: 1.1em; margin-bottom: 20px;'><strong>📋 Enter patient diagnostic measurements to predict tumor classification:</strong></p>", unsafe_allow_html=True)
    
    # Organize inputs in columns for better layout
    st.sidebar.markdown("<h3 style='color: #3370E7; text-align: center;'>📋 INPUT FEATURES</h3>", unsafe_allow_html=True)
    
    # Add clear button in sidebar
    col_clear1, col_clear2 = st.sidebar.columns([1, 1])
    with col_clear1:
        if st.button('🔄 Clear All', use_container_width=True, key='clear_btn'):
            st.session_state.clear()
            st.rerun()
    
    input_data = {}
    
    # Group features by category
    feature_groups = {
        "📏 Mean Measurements": feature_names[0:10],
        "📊 Error Measurements": feature_names[10:20],
        "🔝 Worst Measurements": feature_names[20:30]
    }
    
    # Create expandable sections in sidebar
    for group_name, features in feature_groups.items():
        with st.sidebar.expander(f"{group_name}", expanded=(group_name=="📏 Mean Measurements")):
            for feature in features:
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    value=default_values[feature], 
                    step=0.01,
                    label_visibility="visible",
                    key=f"input_{feature}"
                )
    
    # Convert input data to DataFrame for scaling
    input_df = pd.DataFrame([input_data])
    
    # Prediction button with custom styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button('🚀 Make Prediction', use_container_width=True, key='predict_btn')
    
    if predict_button:
        with st.spinner('🔄 Analyzing data...'):
            # Scale the input data
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction_proba = model.predict(scaled_input, verbose=0)[0][0]
            prediction_class = (prediction_proba >= 0.5).astype(int)
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #3370E7;'>📈 PREDICTION RESULTS</h2>", unsafe_allow_html=True)
        
        # Results display
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction_class == 0:
                st.markdown("""
                    <div class='malignant-result'>
                        <h3 style='color: white; margin: 0;'>⚠️ MALIGNANT</h3>
                        <p style='font-size: 0.9em; margin: 10px 0 0 0;'>High Risk Classification</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<p style='color: #c62828; text-align: center; font-size: 1.05em; margin-top: 15px;'><strong>⚠️ The model predicts that the tumor is <span style=\"color: #ff5252;\">MALIGNANT</span>. Medical consultation is strongly recommended.</strong></p>", unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='benign-result'>
                        <h3 style='color: white; margin: 0;'>✅ BENIGN</h3>
                        <p style='font-size: 0.9em; margin: 10px 0 0 0;'>Low Risk Classification</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<p style='color: #2e7d32; text-align: center; font-size: 1.05em; margin-top: 15px;'><strong>✅ The model predicts that the tumor is <span style=\"color: #4CAF50;\">BENIGN</span>. Regular monitoring is recommended.</strong></p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Confidence Score</div>
                    <div class='metric-value'>{prediction_proba*100:.1f}%</div>
                    <div style='color: #999; font-size: 0.9em;'>Model Certainty</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color: #3370E7; margin-top: 0;'>⚕️ Important Notice</h4>
                    <p style='color: #555; margin: 10px 0;'>This AI prediction is a diagnostic support tool only. 
                    It should not replace professional medical diagnosis. Always consult with qualified healthcare professionals.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color: #3370E7; margin-top: 0;'>🔬 Model Information</h4>
                    <p style='color: #555; margin: 10px 0;'><strong>Type:</strong> Multi-Layer Perceptron (MLP) Neural Network<br>
                    <strong>Framework:</strong> TensorFlow/Keras<br>
                    <strong>Features:</strong> 30 diagnostic measurements</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 style='color: #3370E7;'>📊 Input Data Summary</h3>", unsafe_allow_html=True)
        st.dataframe(input_df, use_container_width=True)

with tab2:
    st.markdown("<h2 style='color: #3370E7;'>📋 Feature Information</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h4 style='color: #3370E7;'>📏 Mean Measurements</h4>
                <p style='color: #555;'>Average values of cell nuclei features including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h4 style='color: #3370E7;'>📊 Error Measurements</h4>
                <p style='color: #555;'>Standard error values for each mean measurement, indicating measurement variability and uncertainty in diagnostic parameters.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='info-box'>
                <h4 style='color: #3370E7;'>🔝 Worst Measurements</h4>
                <p style='color: #555;'>Largest values observed for each feature, representing the most abnormal cell characteristics detected in the sample.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class='info-box'>
            <h4 style='color: #3370E7; margin-top: 0;'>🎯 Classification Categories</h4>
            <p style='color: #555;'><strong>Benign (✅):</strong> Non-cancerous tumors that typically do not pose immediate health risks.<br>
            <strong>Malignant (⚠️):</strong> Cancerous tumors that require medical intervention and treatment.</p>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 style='color: #3370E7;'>ℹ️ About This System</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <h4 style='color: #3370E7; margin-top: 0;'>🏢 Zion Tech Hub AI</h4>
            <p style='color: #555;'>This is an advanced breast cancer detection system developed by Zion Tech Hub, 
            leveraging machine learning to provide accurate and timely diagnostic support for medical professionals.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h4 style='color: #3370E7;'>🤖 Technology Stack</h4>
                <ul style='color: #555;'>
                    <li><strong>ML Framework:</strong> TensorFlow/Keras</li>
                    <li><strong>Model Type:</strong> Neural Network (MLP)</li>
                    <li><strong>Input Features:</strong> 30 diagnostic parameters</li>
                    <li><strong>Accuracy:</strong> High-precision predictions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h4 style='color: #3370E7;'>⚖️ Legal Disclaimer</h4>
                <p style='color: #555; font-size: 0.95em;'>This tool is for educational and diagnostic support purposes only. 
                It is not a substitute for professional medical advice, diagnosis, or treatment. 
                Always consult qualified healthcare professionals for medical decisions.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class='info-box'>
            <h4 style='color: #3370E7; margin-top: 0;'>📞 Support</h4>
            <p style='color: #555;'>For questions or support regarding this system, please contact Zion Tech Hub's medical AI team.</p>
        </div>
    """, unsafe_allow_html=True)
