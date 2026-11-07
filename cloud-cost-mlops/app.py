import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("model/cloud_cost_model.pkl")
app = FastAPI(title="Cloud Infrastructure Cost Prediction API")

class CloudMetrics(BaseModel):
    cpu: float = Field(ge=0, le=100)
    memory: float = Field(ge=0, le=100)
    network: float = Field(ge=0)
    storage: float = Field(ge=0)
    running_hours: float = Field(ge=0, le=744)

@app.get("/")
def home():
    return {"message": "Cloud Cost Prediction API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_cost(data: CloudMetrics):
    values = np.array([[
        data.cpu, data.memory, data.network,
        data.storage, data.running_hours
    ]])
    prediction = float(model.predict(values)[0])
    return {
        "predicted_monthly_cost": round(max(0.0, prediction), 2),
        "currency": "USD",
        "note": "Synthetic teaching estimate, not an actual AWS bill."
    }
