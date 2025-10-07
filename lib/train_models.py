#train_models.py
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset
import firebase_admin
from firebase_admin import credentials, db
import torch
import torch.nn.functional as F
from datetime import datetime

# File management
MODEL_FILES = {
    'predictor': 'water_predictor.pth',
    'anomaly_detector': 'anomaly_detector.pth',
    'scaler': 'scaler.pkl',
    'metrics': 'model_metrics.json'
}

def cleanup_model_files():
    """Remove all existing model files"""
    print("Cleaning up old model files...")
    for file in MODEL_FILES.values():
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")

def verify_model_files():
    """Check if all required model files exist"""
    print("\nVerifying model files...")
    all_exist = True
    for name, path in MODEL_FILES.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'}")
        if not exists:
            all_exist = False
    return all_exist

def setup_firebase():
    """Initialize Firebase connection"""
    config_path = "serviceAccountKey.json"
    if not os.path.exists(config_path):
        print("\nERROR: Missing Firebase credentials")
        exit(1)

    try:
        cred = credentials.Certificate(config_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-water-metering-sys-default-rtdb.firebaseio.com'
        })
    except Exception as e:
        print(f"\nFirebase initialization failed: {e}")
        exit(1)

def prepare_data():
    """Fetch and prepare data from Firebase"""
    try:
        ref = db.reference('sensor_readings')
        data = ref.get()

        if not data:
            print("No data found in Firebase, using synthetic data")
            return generate_synthetic_data()

        records = []
        for key, value in data.items():
            try:
                timestamp = pd.to_datetime(value.get('timestamp'))
                record = {
                    'flow_rate': float(value.get('Flow_rate', 0)),
                    'volume': float(value.get('total_volume', 0)),
                    'hour': timestamp.hour,
                    'weekday': timestamp.weekday(),
                    'month': timestamp.month,
                    'usage': float(value.get('total_volume', 0)) - float(value.get('previous_volume', 0))
                }
                records.append(record)
            except Exception as e:
                print(f"Skipping record {key}: {str(e)}")
                continue

        df = pd.DataFrame(records)
        df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
        return df.dropna()

    except Exception as e:
        print(f"Error fetching data: {e}, using synthetic data")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic data for testing"""
    print("Generating synthetic training data...")
    np.random.seed(42)
    size = 1000
    df = pd.DataFrame({
        'flow_rate': np.random.uniform(0.5, 5.0, size),
        'volume': np.random.uniform(10, 500, size),
        'hour': np.random.randint(0, 24, size),
        'weekday': np.random.randint(0, 7, size),
        'month': np.random.randint(1, 13, size),
        'usage': np.random.uniform(0.1, 10.0, size)
    })
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    return df

class EnhancedUsagePredictor(nn.Module):
    """Improved predictor model with LSTM"""
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(32 + (input_size-6), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        temporal = x[:, :6].unsqueeze(1)
        non_temporal = x[:, 6:]
        _, (hidden, _) = self.lstm(temporal)
        lstm_out = hidden.squeeze(0)
        combined = torch.cat([lstm_out, non_temporal], dim=1)
        return self.dense(combined)

class Autoencoder(nn.Module):
    """Anomaly detection model"""
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_and_save_models():
    """Main training function"""
    cleanup_model_files()

    df = prepare_data()
    X = df[['flow_rate', 'volume', 'hour', 'weekday', 'month', 'is_weekend']].values
    y = df['usage'].values

    # Add historical features placeholder
    X = np.hstack([X, np.random.rand(len(X), 24)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train predictor
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    predictor = EnhancedUsagePredictor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

    print("Training predictor model...")
    for epoch in range(100):
        predictor.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = predictor(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # Train anomaly detector
    anomaly_detector = Autoencoder(6)  # Using core 6 features
    anomaly_train_loader = DataLoader(torch.FloatTensor(X_train[:, :6]), batch_size=32, shuffle=True)

    print("Training anomaly detector...")
    for epoch in range(50):
        anomaly_detector.train()
        for inputs in anomaly_train_loader:
            optimizer.zero_grad()
            outputs = anomaly_detector(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    # Save models
    torch.save(predictor.state_dict(), MODEL_FILES['predictor'])
    torch.save(anomaly_detector.state_dict(), MODEL_FILES['anomaly_detector'])
    joblib.dump(scaler, MODEL_FILES['scaler'])

    # Evaluate and save metrics
    with torch.no_grad():
        predictor.eval()
        predictions = predictor(torch.FloatTensor(X_test))
        test_loss = F.mse_loss(predictions, torch.FloatTensor(y_test).unsqueeze(1))

        ss_res = ((y_test - predictions.numpy().flatten()) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'test_loss': float(test_loss.item()),
            'r2_score': float(r2),
            'mae': float(F.l1_loss(predictions, torch.FloatTensor(y_test).unsqueeze(1)).item()),
            'training_date': datetime.now().isoformat()
        }
        with open(MODEL_FILES['metrics'], 'w') as f:
            json.dump(metrics, f, indent=2)

    print("\nTraining completed successfully!")
    verify_model_files()

if __name__ == "__main__":
    setup_firebase()
    train_and_save_models()