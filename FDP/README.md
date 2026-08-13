# Future Disease Prediction and Prevention System using CNN

**K Balamurugan (212622104018) | G Mahendran (212622104046)**  
Department of Computer Science and Engineering

---

## 📌 Project Overview

A CNN-based Deep Learning system that takes **personal & lifestyle data** as input and predicts the risk of **7 future diseases** along with personalized **prevention strategies**.

### Diseases Predicted
| Disease | Key Risk Factors |
|---------|-----------------|
| Diabetes (Type 2) | BMI, glucose, family history, age |
| Hypertension | BP, stress, smoking, obesity |
| Heart Disease (CVD) | Age, smoking, BP, family history |
| Hypothyroidism | Stress, sleep, age |
| Sleep Disorder | Sleep hours, stress, sedentary job |
| Obesity / Metabolic Risk | BMI, activity, alcohol |
| Occupational Disease | Job type, age, smoking |

---

## 🧠 CNN Architecture

```
Input (16 features)
       ↓
  Reshape → (16, 1)
       ↓
[CONVOLUTION LAYER 1]  Conv1D(32, kernel=3, relu)
       ↓
[POOLING LAYER 1]      MaxPooling1D(pool_size=2)      ← Downsampling
       ↓
[CONVOLUTION LAYER 2]  Conv1D(64, kernel=3, relu)
       ↓
[POOLING LAYER 2]      MaxPooling1D(pool_size=2)      ← Downsampling
       ↓
     Flatten
       ↓
[FULLY CONNECTED 1]    Dense(128, relu) + Dropout(0.3)
       ↓
[FULLY CONNECTED 2]    Dense(64, relu) + Dropout(0.2)
       ↓
[OUTPUT LAYER]         Dense(7, sigmoid)  ← Multi-label output
```

- **Loss:** Binary Cross-Entropy (multi-label)
- **Optimizer:** Adam (lr=0.001)
- **Batch Normalization** after each conv layer
- **Early Stopping** to prevent overfitting

---

## 📊 Input Features (16 total)

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age in years |
| weight | Numeric | Weight in kg |
| height | Numeric | Height in cm |
| bmi | Computed | weight / height² |
| blood_group | Categorical (0–7) | A+, A-, B+, ... |
| job | Categorical (0–14) | 15 job categories |
| sleep_hours | Numeric | Hours of sleep/night |
| stress_level | Numeric (1–10) | Self-reported stress |
| physical_activity_days | Numeric (0–7) | Exercise days/week |
| smoking | Binary | 0=No, 1=Yes |
| alcohol | Binary | 0=No, 1=Yes |
| family_history_diabetes | Binary | 0=No, 1=Yes |
| family_history_heart | Binary | 0=No, 1=Yes |
| systolic_bp | Numeric | Upper BP (mmHg) |
| diastolic_bp | Numeric | Lower BP (mmHg) |
| fasting_glucose | Numeric | Blood glucose (mg/dL) |

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip

### Linux / Mac
```bash
chmod +x run.sh
./run.sh
```

### Windows
```
Double-click run_windows.bat
```

### Manual Step-by-Step
```bash
# Install dependencies
pip install tensorflow flask pandas numpy scikit-learn

# Generate dataset
python generate_dataset.py

# Train CNN model
python train_model.py

# Run web app
python app.py
```

Then open: **http://localhost:5000**

---

## 🌐 Platform Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.8+ |
| RAM | 4 GB |
| Storage | 500 MB |
| GPU | Optional (speeds training) |
| OS | Windows 10 / Ubuntu 18+ / macOS |

---

## 🎯 Expected Accuracy

| Disease | Accuracy |
|---------|----------|
| Diabetes | ~95–97% |
| Hypertension | ~94–96% |
| Heart Disease | ~93–95% |
| Hypothyroidism | ~92–95% |
| Sleep Disorder | ~93–96% |
| Obesity Risk | ~96–98% |
| Occupational Disease | ~93–95% |
| **Overall** | **~95%+** |

---

## 📁 File Structure

```
disease_prediction/
├── app.py                  ← Flask web application
├── train_model.py          ← CNN training script
├── generate_dataset.py     ← Dataset generator
├── predictor.py            ← Prediction engine + prevention advice
├── requirements.txt        ← Python packages
├── run.sh                  ← Linux/Mac setup script
├── run_windows.bat         ← Windows setup script
├── README.md               ← This file
├── data/
│   └── health_dataset.csv  ← Generated training dataset (10,000 rows)
├── models/
│   ├── cnn_disease_model.h5    ← Trained CNN model
│   ├── scaler.pkl              ← Feature scaler
│   ├── features.json           ← Feature/disease metadata
│   ├── accuracy_results.json   ← Per-disease accuracy
│   └── training_history.csv    ← Epoch-wise loss/accuracy
└── templates/
    └── index.html          ← Web UI
```
