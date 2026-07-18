# ==========================
# Step 1: Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Step 2: Load Dataset
# ==========================
df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully!")

# ==========================
# Step 3: Understand Dataset
# ==========================
print("\nFirst 5 Rows:")
print(df.head())

print("\n==========================")
print("Dataset Shape")
print("==========================")
print(df.shape)

print("\n==========================")
print("Dataset Information")
print("==========================")
df.info()

# ==========================
# Step 4: Handle Missing Values
# ==========================

print("\n==========================")
print("Missing Values")
print("==========================")

print(df.isnull().sum())

print("\n==========================")
print("Selected Features")
print("==========================")

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "LotArea"
]

target = "SalePrice"

print("Features:")
print(features)

print("\nTarget Variable:")
print(target)

# ==========================
# Step 6: Create Features & Target
# ==========================

X = df[features]
y = df[target]

print("\nFeatures Dataset (X)")
print(X.head())

print("\nTarget Dataset (y)")
print(y.head())

# ==========================
# Step 7: Split Dataset
# ==========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ==========================
# Step 8: Check Missing Values in Selected Features
# ==========================

print("\n==========================")
print("Missing Values in Selected Features")
print("==========================")

print(X.isnull().sum())

# ==========================
# Step 9: Train Linear Regression Model
# ==========================

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# ==========================
# Step 10: Predict House Prices
# ==========================

predictions = model.predict(X_test)

print("\nPredicted House Prices:")
print(predictions[:10])

# ==========================
# Step 11: Evaluate the Model
# ==========================

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== Model Evaluation ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ==========================
# Step 12: Save Model
# ==========================

import joblib

joblib.dump(model, "house_price_model.pkl")

print("\nModel Saved Successfully!")

# ==========================
# Step 13: Predict One House
# ==========================

sample_house = [[
    7,      # OverallQual
    1710,   # GrLivArea
    2,      # GarageCars
    548,    # GarageArea
    856,    # TotalBsmtSF
    856,    # 1stFlrSF
    2,      # FullBath
    2003,   # YearBuilt
    2003,   # YearRemodAdd
    8450    # LotArea
]]

predicted_price = model.predict(sample_house)

print("\nPredicted Price for Sample House:")
print(predicted_price)