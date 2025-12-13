# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:42:14 2025

@author: Evelyn1
"""

import pandas as pd

url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv"
df = pd.read_csv(url)

df.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv"
df = pd.read_csv(url)

# -------- Splitting Data --------
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -------- Fit Linear Regression Model --------
lr = LinearRegression()
lr.fit(X_train, y_train)

# Get coefficients
intercept = lr.intercept_
coeffs = lr.coef_

# -------- Predictions --------
train_pred = lr.predict(X_train)
test_pred = lr.predict(X_test)

# -------- Performance Metrics --------
# Training
r2_train = r2_score(y_train, train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
mae_train = mean_absolute_error(y_train, train_pred)

# Testing
r2_test = r2_score(y_test, test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
mae_test = mean_absolute_error(y_test, test_pred)


# -------- Results --------

print("Intercept:", round(intercept, 4))
print("TV Coefficient:", round(coeffs[0], 4))
print("Radio Coefficient:", round(coeffs[1], 4))
print("Newspaper Coefficient:", round(coeffs[2], 4))

print("\n--- TRAINING SET METRICS ---")
print("R² Train:", round(r2_train, 4))
print("RMSE Train:", round(rmse_train, 4))
print("MAE Train:", round(mae_train, 4))

print("\n--- TEST SET METRICS ---")
print("R² Test:", round(r2_test, 4))
print("RMSE Test:", round(rmse_test, 4))
print("MAE Test:", round(mae_test, 4))


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


# Polynomial Features 
poly = PolynomialFeatures(degree=5, include_bias=False)

X_train_poly = poly.fit_transform(X_train)      
X_test_poly = poly.transform(X_test)            


# Scale the Polynomial Features
scaler = StandardScaler()

X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)


# Linear Regression 
overfit_model = LinearRegression()
overfit_model.fit(X_train_poly_scaled, y_train)

# Predictions
train_pred_poly = overfit_model.predict(X_train_poly_scaled)
test_pred_poly = overfit_model.predict(X_test_poly_scaled)


# Metrics
r2_train_poly = r2_score(y_train, train_pred_poly)
r2_test_poly = r2_score(y_test, test_pred_poly)

rmse_train_poly = np.sqrt(mean_squared_error(y_train, train_pred_poly))
rmse_test_poly = np.sqrt(mean_squared_error(y_test, test_pred_poly))

mae_train_poly = mean_absolute_error(y_train, train_pred_poly)
mae_test_poly = mean_absolute_error(y_test, test_pred_poly)


# Print results
print(f"Number of polynomial features created: {X_train_poly.shape[1]}")

print("=== POLYNOMIAL MODEL COEFFICIENTS (DEGREE 5) ===")
for i, coef in enumerate(overfit_model.coef_):
    print(f"Feature {i+1}: {coef:.5f}")

print("\nIntercept:", overfit_model.intercept_)

print("\n--- TRAINING SET METRICS ---")
print(f"R² Train: {r2_train_poly:.4f}")
print(f"RMSE Train: {rmse_train_poly:.4f}")
print(f"MAE Train: {mae_train_poly:.4f}")

print("\n--- TEST SET METRICS ---")
print(f"R² Test: {r2_test_poly:.4f}")
print(f"RMSE Test: {rmse_test_poly:.4f}")
print(f"MAE Test: {mae_test_poly:.4f}")




from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- FIT RIDGE MODEL ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly_scaled, y_train)

ridge_train_pred = ridge.predict(X_train_poly_scaled)
ridge_test_pred  = ridge.predict(X_test_poly_scaled)

# --- FIT LASSO MODEL ---
lasso = Lasso(alpha=0.01, max_iter=20000)
lasso.fit(X_train_poly_scaled, y_train)

lasso_train_pred = lasso.predict(X_train_poly_scaled)
lasso_test_pred  = lasso.predict(X_test_poly_scaled)


# METRICS
# ============================================================

# Ridge Metrics
ridge_r2_train  = r2_score(y_train, ridge_train_pred)
ridge_r2_test   = r2_score(y_test,  ridge_test_pred)
ridge_rmse_train = mean_squared_error(y_train, ridge_train_pred)**0.5
ridge_rmse_test  = mean_squared_error(y_test,  ridge_test_pred)**0.5
ridge_mae_train  = mean_absolute_error(y_train, ridge_train_pred)
ridge_mae_test   = mean_absolute_error(y_test,  ridge_test_pred)

# Lasso Metrics
lasso_r2_train  = r2_score(y_train, lasso_train_pred)
lasso_r2_test   = r2_score(y_test,  lasso_test_pred)
lasso_rmse_train = mean_squared_error(y_train, lasso_train_pred)**0.5
lasso_rmse_test  = mean_squared_error(y_test,  lasso_test_pred)**0.5
lasso_mae_train  = mean_absolute_error(y_train, lasso_train_pred)
lasso_mae_test   = mean_absolute_error(y_test,  lasso_test_pred)

# Count zeroed coefficients
lasso_zero_count = sum(lasso.coef_ == 0)
total_coeffs = len(lasso.coef_)

# ============================================================
# PRINT RESULTS
# ============================================================
print("=== RIDGE COEFFICIENTS ===")
for i, c in enumerate(ridge.coef_):
    print(f"Feature {i+1}: {c}")

print("\n=== LASSO COEFFICIENTS ===")
for i, c in enumerate(lasso.coef_):
    print(f"Feature {i+1}: {c}")

print("\n========== RIDGE RESULTS ==========")
print(f"R² Train: {ridge_r2_train:.4f}")
print(f"R² Test:  {ridge_r2_test:.4f}")
print(f"RMSE Train: {ridge_rmse_train:.4f}")
print(f"RMSE Test:  {ridge_rmse_test:.4f}")
print(f"MAE Train: {ridge_mae_train:.4f}")
print(f"MAE Test:  {ridge_mae_test:.4f}")

print("\n========== LASSO RESULTS ==========")
print(f"R² Train: {lasso_r2_train:.4f}")
print(f"R² Test:  {lasso_r2_test:.4f}")
print(f"RMSE Train: {lasso_rmse_train:.4f}")
print(f"RMSE Test:  {lasso_rmse_test:.4f}")
print(f"MAE Train: {lasso_mae_train:.4f}")
print(f"MAE Test:  {lasso_mae_test:.4f}")

print("\n========== LASSO COEFFICIENTS ==========")
print(f"Zero coefficients: {lasso_zero_count} out of {total_coeffs}")



# 1. Get the names of ALL polynomial features
feature_names = poly.get_feature_names_out(X.columns)

# 2. Identify which coefficients belong to the ORIGINAL features (TV, Radio, Newspaper)
original_indices = [i for i, name in enumerate(feature_names) 
                    if name in ["TV", "Radio", "Newspaper"]]

# 3. Print Ridge coefficients for the original 3 features
print("=== RIDGE: Original Feature Coefficients ===")
for idx in original_indices:
    print(f"{feature_names[idx]}: {ridge.coef_[idx]}")

# 4. Print Lasso coefficients for the original 3 features
print("\n=== LASSO: Original Feature Coefficients ===")
for idx in original_indices:
    print(f"{feature_names[idx]}: {lasso.coef_[idx]}")
    
# code to see the 14 remaining features
nonzero_indices = np.where(lasso.coef_ != 0)[0]
print("=== FEATURES LASSO KEPT ===")
for i in nonzero_indices:
    print(feature_names[i], ":", lasso.coef_[i])


