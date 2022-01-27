# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

# +
# Random seeds
np.random.seed(seed=0)  # Set seed for NumPy
random_state = 0

# Generate features, and take norm for use with target
x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=(1000,1))
y = np.sin(x)+np.random.uniform(low=-0.5, high=0.5, size=(1000,1))*2

# Create kernel and define GPR
kernel = RBF() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=random_state)

# Fit GPR model
gpr.fit(x, y)

# Create test data
x_test = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=(1000,1))
y_test = np.sin(x_test)
 
# Predict mean
y_hat, y_sigma = gpr.predict(x_test, return_std=True)

# Initialize plot
fig, ax = plt.subplots(figsize=(10,10))

# Squeeze data
x = np.squeeze(x)
y = np.squeeze(y)
x_test = np.squeeze(x_test)
y_test = np.squeeze(y_test)
y_hat = np.squeeze(y_hat)

# Plot the training data
ax.scatter(x, y, s = 0.5)

# Plot predictive means as blue line
ax.scatter(x_test, y_hat, s = 0.01)
ax.scatter(x_test, y_test, s = 0.01)

# Draw points between the lower and upper confidence bounds
lower = y_hat - y_sigma
upper = y_hat + y_sigma
ax.scatter(x_test, lower, s = 0.01)
ax.scatter(x_test, upper, s = 0.01)
plt.title("GPR Model Predictions")
plt.show()
# -


