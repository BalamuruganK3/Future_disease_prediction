import numpy as np
import pandas as pd
import pickle, os, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

FEATURES = ['age','weight','height','bmi','blood_group','job',
            'sleep_hours','stress_level','physical_activity_days',
            'smoking','alcohol','family_history_diabetes','family_history_heart',
            'systolic_bp','diastolic_bp','fasting_glucose','aqi_category']

DISEASES = ['diabetes','hypertension','heart_disease','hypothyroidism',
            'sleep_disorder','obesity_risk','occupational_disease','respiratory_disease']

os.makedirs('models', exist_ok=True)

def train():
    print("Loading dataset...")
    df = pd.read_csv('data/health_dataset.csv')
    X  = df[FEATURES].values
    y  = df[DISEASES].values

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    with open('models/scaler.pkl','wb') as f: pickle.dump(scaler, f)
    with open('models/features.json','w') as f:
        json.dump({'features': FEATURES, 'diseases': DISEASES}, f)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

    base  = MLPClassifier(hidden_layer_sizes=(256,128,64), activation='relu',
                          solver='adam', max_iter=100, random_state=42,
                          early_stopping=True, validation_fraction=0.1, verbose=False)
    model = MultiOutputClassifier(base, n_jobs=-1)
    print("Training MLP model (this takes 1-2 minutes)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    results = {}
    print("\nAccuracy per disease:")
    for i, d in enumerate(DISEASES):
        acc = accuracy_score(y_test[:,i], y_pred[:,i])
        results[d] = round(acc*100, 2)
        print(f"  {d}: {acc*100:.2f}%")
    overall = accuracy_score(y_test.flatten(), y_pred.flatten())
    print(f"\nOverall: {overall*100:.2f}%")

    with open('models/sklearn_model.pkl','wb') as f: pickle.dump(model, f)
    with open('models/accuracy_results.json','w') as f:
        json.dump({'per_disease': results, 'overall': round(overall*100,2)}, f)
    print("Model saved to models/sklearn_model.pkl")

if __name__ == '__main__':
    train()
