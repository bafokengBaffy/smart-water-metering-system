#api_server.py
from flask import Flask, request, jsonify
import torch
import joblib
import numpy as np
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)

MODEL_FILES = {
    'predictor': 'water_predictor.pth',
    'anomaly_detector': 'anomaly_detector.pth',
    'scaler': 'scaler.pkl',
    'metrics': 'model_metrics.json'
}

class Predictor(torch.nn.Module):
    """Wrapper for the predictor model"""
    def __init__(self, input_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 + (input_size-6), 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        temporal = x[:, :6].unsqueeze(1)
        non_temporal = x[:, 6:]
        _, (hidden, _) = self.lstm(temporal)
        lstm_out = hidden.squeeze(0)
        combined = torch.cat([lstm_out, non_temporal], dim=1)
        return self.dense(combined)

class AnomalyDetector(torch.nn.Module):
    """Wrapper for the anomaly detector"""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 6)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_models():
    """Initialize and load all models"""
    if not all(os.path.exists(f) for f in MODEL_FILES.values()):
        raise RuntimeError("Missing model files. Please train models first.")

    scaler = joblib.load(MODEL_FILES['scaler'])
    input_size = scaler.n_features_in_

    predictor = Predictor(input_size)
    predictor.load_state_dict(torch.load(MODEL_FILES['predictor']))
    predictor.eval()

    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_state_dict(torch.load(MODEL_FILES['anomaly_detector']))
    anomaly_detector.eval()

    return predictor, anomaly_detector, scaler, input_size

try:
    predictor, anomaly_detector, scaler, input_size = load_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Failed to load models: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    data = request.json
    try:
        # Validate input
        required_fields = ['flow_rate', 'volume', 'hour', 'weekday', 'month']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Prepare features
        is_weekend = 1 if data['weekday'] >= 5 else 0
        historical_usage = data.get('historical_usage', [0]*24)

        features = np.array([
            data['flow_rate'],
            data['volume'],
            data['hour'],
            data['weekday'],
            data['month'],
            is_weekend,
            *historical_usage[-24:]
        ]).reshape(1, -1)

        # Validate feature size
        if features.shape[1] != input_size:
            return jsonify({
                'error': f'Expected {input_size} features, got {features.shape[1]}'
            }), 400

        # Make prediction
        features = scaler.transform(features)
        with torch.no_grad():
            prediction = predictor(torch.FloatTensor(features)).item()
            confidence = 0.1 * abs(prediction)

            # Anomaly detection
            core_features = torch.FloatTensor(features[:, :6])
            reconstruction = anomaly_detector(core_features)
            anomaly_score = torch.nn.functional.mse_loss(reconstruction, core_features).item()

            response = {
                'predicted_usage': round(prediction, 2),
                'confidence_interval': [
                    round(max(0, prediction - confidence), 2),
                    round(prediction + confidence, 2)
                ],
                'anomaly_detected': anomaly_score > 0.5,
                'anomaly_score': round(anomaly_score, 4),
                'timestamp': datetime.now().isoformat()
            }

            if 'current_usage' in data:
                response['prediction_error'] = round(data['current_usage'] - prediction, 2)

            return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information and performance"""
    try:
        with open(MODEL_FILES['metrics'], 'r') as f:
            metrics = json.load(f)

        return jsonify({
            'status': 'active',
            'input_size': input_size,
            'performance': metrics,
            'files': {k: os.path.exists(v) for k, v in MODEL_FILES.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)