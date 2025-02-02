import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Simulated training data (you should replace this with your actual dataset)
data = {
    "bedrooms": [3, 4, 5],
    "bathrooms": [2, 3, 4],
    "sqft_living": [1800, 2400, 3000],
    "sqft_lot": [5000, 6000, 7000],
    "floors": [1, 2, 3],
    "waterfront": [0, 1, 0],
    "view": [0, 1, 0],
    "condition": [3, 4, 5],
    "sqft_above": [1500, 2000, 2500],
    "sqft_basement": [300, 400, 500],
    "yr_built": [1995, 2000, 2010],
    "yr_renovated": [2010, 2015, 2020],
    "price": [400000, 600000, 800000]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Features and target variable
X = df.drop("price", axis=1)  # All columns except 'price'
y = df["price"]  # Target variable

# Train the model (Random Forest Regressor)
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Train and save the scaler
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler to the training data

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")

# FastAPI app
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

# Load the trained model and scaler when the FastAPI app starts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Select only the features the model was trained on
        selected_features = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", 
            "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated"
        ]

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

# To run this FastAPI server, use: 
# uvicorn app:app --reload
