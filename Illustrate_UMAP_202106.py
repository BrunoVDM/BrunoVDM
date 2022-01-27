# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import umap
import pandas as pd
from pyDOE2 import *
import matplotlib.pyplot as plt

df = pd.DataFrame(ff2n(10))
reducer = umap.UMAP()
embedding = reducer.fit_transform(df)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP Projection of the Dataset', fontsize=24)


