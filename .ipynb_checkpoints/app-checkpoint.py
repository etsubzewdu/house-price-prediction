import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

# Define the expected input format
class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int

# Define the prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Select only the features the model was trained on
        selected_features = ["bedrooms", "bathrooms", "sqft_living"]  # Update if needed

        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Use only the selected features
        input_data = input_data[selected_features]

        # Transform using the trained scaler
        scaled_input = scaler.transform(input_data)

        # Make prediction
        predicted_price = model.predict(scaled_input)


        return {"predicted_price": float(predicted_price[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
