# Zion Tech Hub AI - Breast Cancer Detection System

An advanced machine learning web application that predicts whether a breast cancer tumor is benign or malignant based on diagnostic measurements. Built with Streamlit and powered by a trained Multi-Layer Perceptron (MLP) neural network.

## Features

- 30 Diagnostic Features - Analyzes mean, error, and worst measurements of cell nuclei
- MLP Neural Network - State-of-the-art machine learning model using TensorFlow/Keras
- Real-time Predictions - Instant tumor classification with confidence scores
- Professional UI - Clean, modern interface with healthcare-focused design
- Easy Testing - Pre-filled default values for quick testing
- Clear Button - Reset all values with one click
- Responsive Design - Works seamlessly on different screen sizes
- Three Info Tabs - Prediction, Feature Information, and About sections

## Quick Start

### Prerequisites
- Python 3.11 or later
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dr-nzube-anthony-anyanwu/Zion_Tech_Hub_AI_Breast_Cancer_Project.git
cd Zion_Tech_Hub_AI_Breast_Cancer_Project
```

2. Create a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## How to Use

1. Open the Prediction Tab - Navigate to the Prediction tab
2. Enter Measurements - Pre-filled default values are ready, or click "Clear All" to enter custom values
3. Make Prediction - Click the "Make Prediction" button
4. View Results - See classification, confidence score, and input data summary

## Input Features (30 total)

- Mean Measurements (10) - Average cell nuclei values
- Error Measurements (10) - Standard error values
- Worst Measurements (10) - Most abnormal cell characteristics

## Model Details

- Model Type: Multi-Layer Perceptron (MLP) Neural Network
- Framework: TensorFlow/Keras
- Input Features: 30 diagnostic measurements
- Output: Binary Classification (Benign/Malignant)
- Files: best_mlp.keras (model), scaler.joblib (scaler)

## Important Disclaimer

This AI system is a diagnostic support tool only and NOT a substitute for professional medical advice. Always consult qualified healthcare professionals.

## About Zion Tech Hub

Zion Tech Hub advances healthcare through innovative AI and machine learning solutions.
Brand Color: #3370E7

## Author

Dr. Nzube Anthony Anyanwu

## License

MIT License

## Links

- GitHub: https://github.com/dr-nzube-anthony-anyanwu/Zion_Tech_Hub_AI_Breast_Cancer_Project
- Dataset: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

---
Last Updated: November 23, 2025
Status: Active & Well-Maintained
