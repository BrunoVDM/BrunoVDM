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

# # Compare_Two_Distributions_Wasserstein_distance_202112

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import wasserstein_distance

# +
# Sample the same distribution twice
# Evaluate the Wasserstein distance
# Evaluate the impact of P(x) on distance
# Evaluate the impact of the size of x

# +
#Set number of random variables n_variables and number of points per variable n_points
n_variables, n_points = 2, 1_000_000

#Generate n_variables, n_points random uniform values
df = pd.DataFrame(np.random.uniform(size=(n_variables,n_points)))
df = pd.DataFrame(norm.ppf(df))
# -

#Plot one random variable distribution and the distribution of the average
plt.hist(df.loc[0,:], bins=100,density=True);
plt.hist(df.loc[1,:], bins=100,density=True, alpha=0.5);
plt.show()

N = [10,20,30,50,100,200,300,500,1000,2000,3000,5000,10_000,20_000,30_000,50_000,100_000,200_000,300_000,500_000,1_000_000]
All_distances=pd.DataFrame([])
for i in range(1000):
    distances=[]
    for n in N:
        #Generate n_variables, n_points random uniform values
        df = pd.DataFrame(np.random.uniform(size=(2,n)))
        df = pd.DataFrame(norm.ppf(df))
        distances.append(wasserstein_distance(df.loc[0,:], df.loc[1,:]))
    All_distances=pd.concat([All_distances,pd.DataFrame(distances).T])

fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(visible=True, which='both')
plt.scatter(N, All_distances.mean())


