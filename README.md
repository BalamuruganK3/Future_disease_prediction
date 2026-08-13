# Future Disease Prediction and Prevention System

An AI-powered healthcare prediction platform that uses **deep learning (1D Convolutional Neural Networks)** to predict the risk of multiple chronic diseases based on lifestyle, physiological, and family-history data. The system provides **early risk assessment, preventive recommendations, and health analytics** through a user-friendly web interface.

## Overview

The Future Disease Prediction and Prevention System is designed to identify potential chronic disease risks before symptoms become severe. By analyzing lifestyle patterns and health indicators, the model predicts disease probabilities and helps users take preventive action through personalized recommendations.

The platform focuses on **early detection, prevention, and health awareness**, making it useful for individuals, healthcare awareness programs, and preventive health screening applications.

## Key features

* Predicts risk for **8 chronic diseases**
* Deep learning model using **1D Convolutional Neural Networks**
* Lifestyle and physiological data analysis
* Real-time prediction interface
* Risk categorization (Low / Medium / High)
* Personalized prevention recommendations
* PDF health report generation
* Fast inference with low latency
* Responsive web application

## Diseases predicted

* Diabetes
* Hypertension
* Heart Disease
* Hypothyroidism
* Sleep Disorder
* Obesity Risk
* Occupational Disease
* Respiratory Disease

## Input features

The model analyzes **16 health and lifestyle parameters**:

* Age
* Weight
* Height
* BMI (calculated)
* Blood Group
* Occupation
* Sleep Hours
* Stress Level
* Physical Activity
* Smoking Status
* Alcohol Consumption
* Family History of Diabetes
* Family History of Heart Disease
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Fasting Blood Glucose
* Air Quality Index (AQI)

## Technology stack

| Category            | Technology            |
| ------------------- | --------------------- |
| AI Model            | TensorFlow / Keras    |
| Deep Learning       | 1D CNN                |
| Backend             | Flask                 |
| Frontend            | HTML, CSS, JavaScript |
| Data Processing     | NumPy, Pandas         |
| Machine Learning    | Scikit-learn          |
| Model Serialization | Joblib                |
| Report Generation   | jsPDF                 |

## Model architecture

The prediction engine is built using a **1D Convolutional Neural Network**.

```text
Input Features (16)
        │
        ▼
Conv1D (32 filters)
        │
        ▼
MaxPooling1D
        │
        ▼
Conv1D (64 filters)
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer
        │
        ▼
Dropout
        │
        ▼
Output Layer (8 Disease Probabilities)
```

## Dataset

* **10,000 synthetic health records**
* Medically correlated feature generation
* Multi-label disease classification
* 80/20 training-test split
* Feature normalization using StandardScaler

## Project structure

```text
Future_disease_prediction/
│
├── app.py
├── model/
│   ├── disease_model.h5
│   ├── scaler.pkl
│   ├── features.json
│   └── sklearn_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── dataset/
│
├── utils/
│
├── README.md
└── requirements.txt
```

## Installation

### Clone the repository

```bash
git clone https://github.com/BalamuruganK3/Future_disease_prediction.git
cd Future_disease_prediction
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\\Scripts\\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## API endpoints

| Endpoint  | Method | Description                  |
| --------- | ------ | ---------------------------- |
| /predict  | POST   | Generate disease predictions |
| /ready    | GET    | Model readiness check        |
| /accuracy | GET    | Model performance metrics    |

## Risk classification

| Probability    | Risk Level |
| -------------- | ---------- |
| Less than 0.35 | Low        |
| 0.35 to 0.59   | Medium     |
| 0.60 and above | High       |

## Performance

| Metric                  | Value                |
| ----------------------- | -------------------- |
| Overall Accuracy        | 90.24%               |
| Mean Prediction Latency | 34 ms                |
| 95th Percentile Latency | 47 ms                |
| Maximum Latency         | 83 ms                |
| Memory Usage            | Approximately 180 MB |

## Future enhancements

* Electronic Health Record integration
* Wearable device support
* Real medical dataset training
* Explainable AI (SHAP / LIME)
* Doctor consultation integration
* Cloud deployment
* Multi-language support
* Continuous health monitoring
* Personalized nutrition planning

## Research significance

This project demonstrates the application of **deep learning in preventive healthcare**, emphasizing:

* Early disease detection
* Lifestyle-based health prediction
* Multi-label medical classification
* Low-latency AI inference
* Explainable preventive recommendations
