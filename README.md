

```markdown
# 🏡 Predicting House Prices Using Machine Learning

This project predicts house prices based on property features using a regression model. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment via FastAPI for real-time predictions. 🌟

---

## Features 🛠️

The dataset includes the following features:  
- `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated` (numerical features).  
- `price`: Target variable (house price). 💰

---

## Technologies Used 💻

- **Python:** pandas, numpy, matplotlib, seaborn, scikit-learn, joblib.  
- **API Framework:** FastAPI 🚀.  

---

## Installation and Usage 📝

1. Clone the repository:  
   ```bash
   git clone https://github.com/etsubzewdu/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:  
   ```bash
   python app.py
   ```

4. Start the API:  
   ```bash
   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Use the API for predictions https://house-price-prediction-15.onrender.com for interactive documentation). 📜

---

## Example API Request 📡

**POST** `/predict`  

Input:  
```json
{
  "date": "2025-02-01",
  "price": 500000,
  "bedrooms": 3.0,
  "bathrooms": 2.0,
  "sqft_living": 1800.0,
  "sqft_lot": 5000.0,
  "floors": 2.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "sqft_above": 1500.0,
  "sqft_basement": 300.0,
  "yr_built": 1995,
  "yr_renovated": 2010,
  "street": "123 Elm St",
  "city": "Seattle",
  "statezip": "WA 98101",
  "country": "USA"
}
```



---

## Results 📊

- **Model:** Linear Regression.  
- **Metric:** Mean Squared Error (MSE): `X.XX`.  

The model captures trends but could be enhanced with advanced regression models and larger datasets. 📈

---

## Future Improvements 🚧

- Add location-based features (e.g., neighborhood ratings).  
- Use advanced models like Random Forest or Gradient Boosting. 🌲  
- Train on real-world datasets for better generalization. 🌍

---

## License 📝

Licensed under the MIT License. Modify and use as needed.  
```

