"""
CNN Model for Future Disease Prediction
Architecture: Input -> Conv1D -> MaxPool -> Conv1D -> MaxPool -> Flatten -> FC -> FC -> Output
"""

import numpy as np
import pandas as pd
import pickle, os, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

DISEASES = ['diabetes', 'hypertension', 'heart_disease', 'hypothyroidism',
            'sleep_disorder', 'obesity_risk', 'occupational_disease',
            'respiratory_disease']

FEATURES = ['age', 'weight', 'height', 'bmi', 'blood_group', 'job',
            'sleep_hours', 'stress_level', 'physical_activity_days',
            'smoking', 'alcohol', 'family_history_diabetes',
            'family_history_heart', 'systolic_bp', 'diastolic_bp',
            'fasting_glucose', 'aqi_category']

os.makedirs('models', exist_ok=True)

def build_cnn(input_dim, num_classes):
    """
    CNN Architecture:
    Input (16 features) -> Reshape to (16,1) for Conv1D
    -> Conv1D(32, kernel=3) [Convolution Layer]
    -> MaxPooling1D(2)      [Pooling / Downsampling Layer]
    -> Conv1D(64, kernel=3) [Second Convolution Layer]
    -> MaxPooling1D(2)      [Second Pooling / Downsampling]
    -> Flatten
    -> Dense(128)           [Fully Connected Layer 1]
    -> Dropout(0.3)
    -> Dense(64)            [Fully Connected Layer 2]
    -> Dense(num_classes, sigmoid) [Output Layer]
    """
    model = Sequential([
        Reshape((input_dim, 1), input_shape=(input_dim,)),

        # --- Convolution Layer 1 ---
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv1'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, name='pool1'),    # Pooling / Downsampling

        # --- Convolution Layer 2 ---
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv2'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, name='pool2'),    # Pooling / Downsampling

        Flatten(name='flatten'),

        # --- Fully Connected Layer 1 ---
        Dense(128, activation='relu', name='fc1'),
        Dropout(0.3),

        # --- Fully Connected Layer 2 ---
        Dense(64, activation='relu', name='fc2'),
        Dropout(0.2),

        # --- Output Layer (multi-label) ---
        Dense(num_classes, activation='sigmoid', name='output'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train():
    print("Loading dataset...")
    df = pd.read_csv('data/health_dataset.csv')

    X = df[FEATURES].values
    y = df[DISEASES].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and metadata
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/features.json', 'w') as f:
        json.dump({'features': FEATURES, 'diseases': DISEASES}, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_cnn(len(FEATURES), len(DISEASES))
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint('models/cnn_disease_model.h5', save_best_only=True, monitor='val_loss')
    ]

    print("\nTraining CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n=== Test Set Evaluation ===")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    overall_acc = accuracy_score(y_test.flatten(), y_pred.flatten())
    print(f"Overall accuracy: {overall_acc * 100:.2f}%")

    results = {}
    for i, disease in enumerate(DISEASES):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        results[disease] = round(acc * 100, 2)
        print(f"  {disease}: {acc*100:.2f}%")

    with open('models/accuracy_results.json', 'w') as f:
        json.dump({'per_disease': results, 'overall': round(overall_acc*100, 2)}, f)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('models/training_history.csv', index=False)

    print("\nModel saved to models/cnn_disease_model.h5")
    return model, history


if __name__ == '__main__':
    train()
