# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Illustrate_Central_Limit_Theorem_202112

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# +
#Set number of random variables n_variables and number of points per variable n_points
n_variables, n_points = 100, 1_000_000

#Generate n_variables, n_points random uniform values
df = pd.DataFrame(np.random.uniform(size=(n_variables,n_points)))

#Generate n_variables random uniform values to change magnitude of each random variable
sr = pd.Series(np.random.uniform(size=(n_variables)))

#Change each random variable accordingly
df = df.mul(sr, axis = 0)

# Fit a normal distribution to the data:
mu, std = norm.fit(df.mean())

#Plot one random variable distribution and the distribution of the average
plt.hist(df.loc[0,:], bins=100,density=True);
plt.show()
plt.hist(df.mean(), bins=100, density=True);

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()
# -

