#ml_server.py
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict
import torch
import os
import uvicorn
from datetime import datetime
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Water Metering API",
    description="API for water usage prediction and anomaly detection",
    version="3.0"
)

# Constants
MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILES = {
    'predictor': MODEL_DIR / 'water_predictor.pth',
    'anomaly_detector': MODEL_DIR / 'anomaly_detector.pth',
    'scaler': MODEL_DIR / 'scaler.pkl'
}

# Model Definitions with consistent architecture
class WaterPredictor(nn.Module):
    def __init__(self, input_size=30):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

class AnomalyDetector(nn.Module):
    def __init__(self, input_size=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def initialize_models():
    """Initialize models with proper error handling"""
    try:
        logger.info("Initializing models...")

        # Clean up old model files if they exist
        for model_file in MODEL_FILES.values():
            if model_file.exists():
                os.remove(model_file)
                logger.info(f"Removed old {model_file.name}")

        # Initialize new scaler
        logger.info("Creating new scaler")
        scaler = StandardScaler()
        dummy_data = np.random.rand(100, 30)  # 6 core + 24 historical
        scaler.fit(dummy_data)
        joblib.dump(scaler, MODEL_FILES['scaler'])

        # Initialize and save predictor
        logger.info("Creating new predictor")
        predictor = WaterPredictor()
        torch.save(predictor.state_dict(), MODEL_FILES['predictor'])

        # Initialize and save anomaly detector
        logger.info("Creating new anomaly detector")
        anomaly_detector = AnomalyDetector()
        torch.save(anomaly_detector.state_dict(), MODEL_FILES['anomaly_detector'])

        return predictor, anomaly_detector, scaler

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

# Initialize components
try:
    predictor, anomaly_detector, scaler = initialize_models()
    logger.info("All models initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize: {e}")
    sys.exit(1)

# API Models
class PredictionRequest(BaseModel):
    flow_rate: float
    volume: float
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    month: int        # 1-12
    historical_usage: List[float]
    current_usage: Optional[float] = None

class PredictionResponse(BaseModel):
    predicted_usage: float
    confidence_interval: List[float]
    anomaly_score: float
    anomaly_detected: bool
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Prepare features
        is_weekend = 1 if request.day_of_week >= 5 else 0
        features = np.array([
            request.flow_rate,
            request.volume,
            request.time_of_day,
            request.day_of_week,
            request.month,
            is_weekend,
            *request.historical_usage[-24:]  # Last 24 hours
        ]).reshape(1, -1)

        # Validate input size
        if features.shape[1] != 30:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 30 features, got {features.shape[1]}"
            )

        # Scale features
        features = scaler.transform(features)

        with torch.no_grad():
            # Predict usage
            prediction = predictor(torch.FloatTensor(features)).item()

            # Calculate confidence
            confidence = 0.1 * abs(prediction)

            # Anomaly detection
            core_features = torch.FloatTensor(features[:, :6])
            reconstruction = anomaly_detector(core_features)
            anomaly_score = float(nn.MSELoss()(reconstruction, core_features))

            return {
                "predicted_usage": round(prediction, 4),
                "confidence_interval": [
                    round(max(0, prediction - confidence), 4),
                    round(prediction + confidence, 4)
                ],
                "anomaly_score": round(anomaly_score, 4),
                "anomaly_detected": anomaly_score > 0.5,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(port=8001):
    """Start the server with port handling"""
    try:
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except OSError as e:
        if "10013" in str(e):
            logger.error(f"Port {port} is in use. Trying alternative port...")
            start_server(port + 1)
        else:
            logger.error(f"Server error: {e}")
            raise

if __name__ == "__main__":
    start_server()