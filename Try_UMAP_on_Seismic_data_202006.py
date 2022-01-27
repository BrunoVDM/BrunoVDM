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

import csv
import os
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import autokeras as ak
#from random import choices
# %matplotlib inline
import time
import warnings
import random
warnings.filterwarnings("ignore")
from tabulate import tabulate

# +
listOfFiles = os.listdir('E:\Bruno_20200618')
len(listOfFiles)
df = []
df = pd.DataFrame(df)

for filename in listOfFiles:
    file_path=os.path.join('E:\Bruno_20200618', filename)
    data = pd.read_csv(file_path, header=None,delimiter=' ')   
    data=pd.DataFrame(data)
    df=pd.concat([df, data], axis=1)

df_unique=df.drop_duplicates()
scaler = StandardScaler()
df_unique_Standardized=scaler.fit_transform(df_unique)
df_unique_Standardized=pd.DataFrame(df_unique_Standardized)

#df_Standardized=scaler.transform(df)

# +
#df_unique_Standardized_dropped = df_unique_Standardized.drop(df_unique_Standardized.columns[0], axis=1)
#df_unique_Standardized_dropped.shape
#df_unique_Standardized=pd.DataFrame(df_unique_Standardized)
#df_unique_Standardized

# +
Bla=[]
Times=[]
data_shuffled = df_unique_Standardized.reindex(np.random.permutation(df_unique_Standardized.index))
data_shuffled.index = df_unique_Standardized.index

for i in range(1):
    tic = time.time()
    x_train = data_shuffled.drop(data_shuffled.columns[i], axis=1)
    y_train = data_shuffled.iloc[:][i]
# It tries 10 different models.
    reg = ak.StructuredDataRegressor(overwrite=True)
# Feed the structured data regressor with training data.
    reg.fit(x_train, y_train)
    Bla.append(reg.evaluate(x_train, y_train))
# Predict with the best model.
#predicted_y = reg.predict(x_test)
# Evaluate the best model with testing data.
#print(reg.evaluate(x_test, y_test))
    toc = time.time()
    Times.append(toc-tic)
    print(i+1,toc-tic)
# -

data_shuffled
#df_unique_Standardized
#print(reg.evaluate(x_train, y_train))
#Bla=[]
#Bla.append(reg.evaluate(x_train, y_train))
#pd.DataFrame(Bla).to_csv('BLA_20200726.csv',index=False)

steps = [30,100,300,1000,3000,10000,30000,100000,300000] #,1000000]
for step in steps:
    tic = time.time()
    reducer = umap.UMAP();
    embedding = reducer.fit_transform(pd.DataFrame(df_unique_Standardized).sample(n=step,random_state=42));
    toc = time.time()
    print(step, toc-tic)

df_embedding = reducer.transform(pd.DataFrame(df_unique_Standardized).sample(n=step,random_state=42));
plt.scatter(df_embedding[:, 0], df_embedding[:, 1],s=0.0001)#,alpha=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the overall dataset', fontsize=24);

df_embedding = reducer.transform(df_unique_Standardized);
plt.scatter(df_embedding[:, 0], df_embedding[:, 1],s=0.0001)#,alpha=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the overall dataset', fontsize=24);

plt.scatter(df_embedding[:, 0], df_embedding[:, 1],s=0.00001,alpha=0.3)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the overall dataset', fontsize=24);

# +
tic = time.time()
df_embedding_data_unique_Standardized=pd.concat([pd.DataFrame(df_unique_Standardized), pd.DataFrame(df_embedding)], axis=1)
toc = time.time()
print(toc-tic)
df_embedding_data_unique_Standardized.columns = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 
                                                 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 
                                                 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 
                                                 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 
                                                 'x', 'y']

df_Standardized=pd.DataFrame(df_Standardized)
df_Standardized.columns = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 
                           'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 
                           'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 
                           'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39']
# -

tic = time.time()
df_final = pd.merge(df_Standardized, df_embedding_data_unique_Standardized, 
                    on=['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 
                        'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 
                        'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 
                        'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39'], 
                    how='inner')
df_final.describe()
#toc = time.time()

tic = time.time()
#df_final.to_csv('E:/Results.csv', index=False)
df_final.filter(items=['x']).to_csv('E:/x_20200619.csv', index=False)
df_final.filter(items=['y']).to_csv('E:/y_20200619.csv', index=False)
toc = time.time()
print(toc-tic)

csvfile1 = 'E:/csvfile1.csv'
data1 = pd.read_csv(csvfile1, header=None) #, delimiter=' ', quotechar='|')
csvfile2 = 'E:/csvfile2.csv'
data2 = pd.read_csv(csvfile2, header=None) #, delimiter=' ', quotechar='|')
csvfile3 = 'E:/csvfile3.csv'
data3 = pd.read_csv(csvfile3, header=None) #, delimiter=' ', quotechar='|')
data=pd.concat([data1, data2,data3], axis=1)
#random.seed(42)
#3340898 total, 441380 unique
data_unique=data.drop_duplicates()
#data_shuffled = data.reindex(np.random.permutation(data.index))
scaler = StandardScaler()
data_unique_Standardized=scaler.fit_transform(data_unique)
data_Standardized=scaler.transform(data)

df_unique_Standardized.shape

df=pd.DataFrame(data_unique_Standardized)
df.describe()

df=pd.DataFrame(data_Standardized)
df.describe()

tic = time.time()
reducer = umap.UMAP();
embedding = reducer.fit_transform(data_unique_Standardized);
toc = time.time()
print(toc-tic)

df_embedding = reducer.transform(data_Standardized);
plt.scatter(data_embedding[:, 0], data_embedding[:, 1],s=0.0001)#,alpha=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the overall dataset', fontsize=24);

# +
tic = time.time()
embedding_data_unique_Standardized=pd.concat([pd.DataFrame(data_unique_Standardized), pd.DataFrame(embedding)], axis=1)
toc = time.time()
print(toc-tic)
embedding_data_unique_Standardized.columns = ['AZ1','AZ2','AZ3',
                     'x','y']

data_Standardized=pd.DataFrame(data_Standardized)
data_Standardized.columns = ['AZ1','AZ2','AZ3']

# -

tic = time.time()
df_final = pd.merge(data_Standardized, embedding_data_unique_Standardized, 
                    on=['AZ1','AZ2','AZ3'], how='inner')
df_final.describe()
#toc = time.time()

#df_final
#for alpha in (0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1):
plt.scatter(df_final['x'], df_final['y'],c=df_final['AZ3'])#,alpha=0.1)
plt.show()
#plt.scatter(df_final['x'], df_final['y'],s=0.001)#,alpha=0.1)
#plt.gca().set_aspect('equal', 'datalim')
#plt.title('UMAP projection of the overall dataset', fontsize=24);

df.filter(items=['x', 'y'])

tic = time.time()
#df_final.to_csv('E:/Results.csv', index=False)
df_final.filter(items=['x', 'y']).to_csv('E:/Results_02.csv', index=False)
toc = time.time()
print(toc-tic)


plt.scatter(data_embedding[:, 0], data_embedding[:, 1],s=0.0001)#,alpha=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the overall dataset', fontsize=24);

df=pd.DataFrame(data_unique_Standardized)
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
sns.pairplot(df, diag_kind="kde", markers="+", kind="reg")

sizes = [100,300,1*10**3,3*10**3,1*10**4,3*10**4,1*10**5,441380]
reducer = umap.UMAP()
#embeddings=[]
for size in sizes:
    tic = time.time()
    embedding = reducer.fit_transform(data_unique_Standardized[:size]);
    #plt.scatter(embedding[:, 0], embedding[:, 1],s=1,alpha=0.1)
    #plt.gca().set_aspect('equal', 'datalim')
    #plt.title('UMAP projection of the dummy dataset', fontsize=24);
    toc = time.time()
    print(size, toc-tic)

# +

tic = time.time()
plt.scatter(embedding[:, 0], embedding[:, 1],s=0.0001)#,alpha=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dummy dataset', fontsize=24);
toc = time.time()
print(size, toc-tic)


# -

def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data_unique_Standardized);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)),s=0.001)#, c=data_unique)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1],s=0.001)#, c=data_unique)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2],s=0.001)#, c=data, s=100)
    plt.title(title, fontsize=18)


for n in (3, 10, 30, 100, 300):
    #tic = time.time()
    draw_umap(n_neighbors=n, title='n_neighbors = {}'.format(n))
    #toc = time.time()
    #times+= toc-tic
    #print(size, toc-tic)

for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99):
    draw_umap(min_dist=d, title='min_dist = {}'.format(d))

for m in ("euclidean","manhattan","chebyshev","minkowski",
          "canberra","braycurtis","mahalanobis",
          "seuclidean","cosine","correlation"):
    name = m if type(m) is str else m.__name__
    draw_umap(n_components=2, metric=m, title='metric = {}'.format(name))

# +
# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause


from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

# Run the fit
clust.fit(X)

labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()
# -






