# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Sample Data
data = {
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 1, 4, 3, 5],
    'Y':  [3, 4, 6, 8, 10]
}
# Create DataFrame
df = pd.DataFrame(data)

# Independent and Dependent variables
X1 = df[['X1']]                
X_multi = df[['X1', 'X2']]     
Y = df['Y']

# -------------------------------
# 1. Simple Linear Regression
# -------------------------------
model_s = LinearRegression()
model_s.fit(X1, Y)
pred_s = model_s.predict(X1)

r2_s = round(r2_score(Y, pred_s), 2)
mae_s = round(mean_absolute_error(Y, pred_s), 2)
rmse_s = round(np.sqrt(mean_squared_error(Y, pred_s)), 2)

# -------------------------------
# 2. Multiple Linear Regression
# -------------------------------
model_m = LinearRegression()
model_m.fit(X_multi, Y)
pred_m = model_m.predict(X_multi)

r2_m = round(r2_score(Y, pred_m), 2)
mae_m = round(mean_absolute_error(Y, pred_m), 2)
rmse_m = round(np.sqrt(mean_squared_error(Y, pred_m)), 2)

# -------------------------------
# 3. Polynomial Regression (Degree = 2)
# -------------------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X1)

model_p = LinearRegression()
model_p.fit(X_poly, Y)
pred_p = model_p.predict(X_poly)

r2_p = round(r2_score(Y, pred_p), 2)
mae_p = round(mean_absolute_error(Y, pred_p), 2)
rmse_p = round(np.sqrt(mean_squared_error(Y, pred_p)), 2)

print("\n{:<35} {:<10} {:<10} {:<10}".format("Regression Model", "R² Score", "MAE", "RMSE"))
print("-" * 65)

print("{:<35} {:<10} {:<10} {:<10}".format("Simple Linear Regression", r2_s, mae_s, rmse_s))
print("{:<35} {:<10} {:<10} {:<10}".format("Multiple Linear Regression", r2_m, mae_m, rmse_m))
print("{:<35} {:<10} {:<10} {:<10}".format("Polynomial Regression (Degree = 2)", r2_p, mae_p, rmse_p))