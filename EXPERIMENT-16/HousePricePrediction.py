# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("C:/Users/dell/Desktop/804-B/MLOPS/EXPERIMENT-16/House Price Prediction Dataset.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

X = data.drop(columns=["Price", "Id"])
y = data["Price"]

categorical_features = [
    "Location",
    "Condition",
    "Garage"
]

numerical_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model.fit(X_train, y_train)

print("\nModel training completed.")

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))

new_house = pd.DataFrame({
    "Area": [3000],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Floors": [2],
    "YearBuilt": [2015],
    "Location": ["Suburban"],
    "Condition": ["Good"],
    "Garage": ["Yes"]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(round(predicted_price[0], 2))