# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:42:14 2025

@author: Evelyn1
"""

import pandas as pd

url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv"
df = pd.read_csv(url)

df.head()


# ============================================
# PART 1 — The “Simple” Model (Baseline)
# ============================================

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


# ============================================
# RESULTS
# ============================================

# Intercept: 2.9372
# TV Coefficient: 0.0470
# Radio Coefficient: 0.1766
# Newspaper Coefficient: 0.0019

# --- Training Set Performance ---
# R² Train: 0.8850          
# RMSE Train: 1.7897        
# MAE Train: 1.3747        

# --- Test Set Performance ---
# R² Test: 0.9225           
# RMSE Test: 1.3889         
# MAE Test: 1.0548          



# ============================================================
# PART 2 — THE OVERLY COMPLEX MODEL (THE TRAP)
# ============================================================

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



# ============================================================
# Question 1
# ============================================================


# What do you observe about the coefficients?

# The coefficients are very large as some are either very positive or very negative and almost none 
# are near zero. The signs also flip constantly with no economic or intuitive pattern to the switching signs.
# This lack of intuition is a classic sign of overfitting because the model is just memorising noise.
# When the model does this, it forces the curve to bend unnaturally by assigning high and unstable weights 
# to high degree terms. There is no economic meaning behind it and the model just uses them to fit noise perfectly.
# Therefore, I see it as a high risk model because it overfits very aggresively with large, non-intuitive
# and unstable coefficients.

# Print the metrics for training set

# RMSE Train: 0.2493       
# MAE Train: 0.1940  
# R² Train: 0.9978  
       
# Print R² for test set

# R² Test: 0.7944    


# What do you observe with the metrcis

# The first thing that stands out to me is how good the values seem to be in the polynomial model as the R² is very
# high and there are very small errors shown in the RMSE and MAE. These values indicate that the model fits the 
# training data almost perfectly. In real datasets, these tiny errors and very high R² are not good signs and they
# dont indicqte that the model is amazing. Rather, it is signs of the model memorising the training data, including 
# its randomness and noise.
# If we compare the training and test results, we can see that the R² drops to 0.79 and the RMSE and MAE rise for the
# test set. This gap between the values for the training and test set shows that the model has indeed overfit.
# Essentially, the model has learned the training data very specifically and even learned non-generalisable data, so
# the patterns are not real it is just left over noise in the training set. An actual good model would perform similarly
# on training and test data.
# In part 1, the R² was 0.92 which is better than 0.79 that we get from the polynomial model. So it tells us that being
# more complex doesn't mean the model will perform better and it has actually made it worse here. So the simple model
# is better at predicting new data even though it doesnt fit the data in a perfect way, like the polynomial model did.
# It is not perfect because it is trying to capture the true underlying relationships instead of just forcing things to fit.
# So in conclusion, the complex polynomial model is not a good model and just looks impressive with the training data.
# It has large and unstable coefficients, with wide differences in the metrics for the test and train data, showing that
# the simpler model was better. A good model should be stable and perform well on new data and not just data its seen.





# ============================================
# PART 3 — THE REGULARIZATION FIX (RIDGE & LASSO)
# ============================================

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




# ============================================
# Question 2
# ============================================

# What do you observe about the coefficients?

# The coefficients in the ridge and lasso models are extremely smaller and more stable than in the polynomial model.
# Ridge manages to shrink all the coefficients towards 0 but they stay as being non-zero. This is good because it
# smooths the model and prevents extreme swings. Lasso not only shrinks them but completely removes many of them so
# they go entirely to 0. So the coefficients now are more stable, intuitive and consistent with what we would 
# expect from real marketing data, so its more interpretable. 


# How many features from the lasso model was set to zero.

# The lasso model set 41 out of the 55 polynomial features to zero. This shows that those high-degree polynomial 
# interactions in part 2 were not actually providing any meaningful information because for lasso to remove a feature
# it is essentiall stating that this feature does not help prediction. So by removing 41, we see that the data is driven
# by a smaller and simpler set of relationships. Newspaper variables and most polynomial expansions did not provide
# any real predictive power. So the 14 left are the true underlying drivers of sales.


# Performance metrics for ridge model.

# R² Train: 0.9855
# R² Test:  0.9927
# RMSE Train: 0.6355
# RMSE Test:  0.4259
# MAE Train: 0.4047
# MAE Test:  0.3152

# Performance metrics for lasso model.

# R² Train: 0.9871
# R² Test:  0.9941
# RMSE Train: 0.5995
# RMSE Test:  0.3842
# MAE Train: 0.3828
# MAE Test:  0.2983


# How the metrics compare to the overfit models test score.

# Compared to the overfit model, both ridge and lasso show big improvements. Their R² is much higher than the 0.79 of
# the overfit model. This big difference shows how risky overfitting is because even though its training R² was nearly
# perfect, it failed on unseen data. So regularisation methods by controlling coefficient size in the ridge model
# or removing unncessary features in the lasso model, led to better prediction and generalisation. Thus, comparing
# their scores shows that regularisation is essential or models would be unstable, inaccurate and un-intuitive.



# ============================================
# Question 3
# ============================================

# After the four models, TV and Radio seem to be the best. In the simple model in part 1, TV and Radio had the highest
# coefficients.
# In the Ridge model, TV has the strongest coefficient so increases in TV spending consistently lead to increases in 
# sales. Radio also has a positive impact, although smaller than TV. However, newspaper barely affects sakes at all 
# as the coefficient is close to zero.
# In the lasso model, we can see that TV dominates in the list of features that lasso kept and radio also has a 
# meaningful contribution but newspaper barely contributes. Again reinforcing the conclusion that its TV and Radio
# which show the true signals. 
# This is important especially since ridge and lasso had the best test performance so they are stable and able to
# generalise well to new data. So overall, TV is the strongest and most consistent channel to affect sales and has the 
# highest impact, while radio is second and newspaper barely provies any help.
# So I would recommend the CMO to invest primarily in TV and maybe even reduce and reallocate funding in newspaper.
