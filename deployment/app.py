from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


app = FastAPI(title="Champion Model Inference API")


class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Champion model inference API is running."}

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])[0]
    return {"prediction": int(prediction)}