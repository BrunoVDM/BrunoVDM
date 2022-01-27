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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas_profiling

# +
# Record start time
start_time = time.time()

# Import files
df01 = pd.read_excel('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD01_Actifs2.xlsx', engine='openpyxl')
df02 = pd.read_excel ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD02_Souscriptions.xlsx', engine='openpyxl')
df03 = pd.read_excel ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD03_Premium.xlsx', engine='openpyxl')
df04 = pd.read_excel ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD04_Conso.xlsx', engine='openpyxl')
df05 = pd.read_csv ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD05_Consent.txt', sep=";")
df06 = pd.read_csv ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD06_Rejets.txt', sep=";")
#df07 = pd.read_csv ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/R_Campagne.xlsx')
#df08 = pd.read_csv ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/R_CP-idOffre.txt')
df09 = pd.read_excel ('C:/Users/van_d/Documents/DATA_CHALLENGE_2019/Base_A/CD09_Parking_2018.xlsx', engine='openpyxl')

# Compute elapsed time
elapsed_time = time.time() - start_time

# Print elapsed time
print(elapsed_time)
# -

df01.info()
#df02.info()
#df02.info()
#df03.info()
#df04.info()
#df05.info()
#df06.info()
#df09.info()

# +
# Record start time
#start_time = time.time()

# Generate report
#pandas_profiling.ProfileReport(df01)

# Compute elapsed time
#elapsed_time = time.time() - start_time

# Print elapsed time
#print(elapsed_time)

# +
# Record start time
#start_time = time.time()

# Generate report
pandas_profiling.ProfileReport(df02)

# Compute elapsed time
#elapsed_time = time.time() - start_time

# Print elapsed time
#print(elapsed_time)

# +
# Record start time
start_time = time.time()

# Drop duplicate rows and rows with missing values
df01=df01.drop_duplicates().dropna()
df02=df02.drop_duplicates().dropna()
df03=df03.drop_duplicates().dropna()
df04=df04.drop_duplicates().dropna()
df05=df05.drop_duplicates().dropna()
df06=df06.drop_duplicates().dropna()
#df07=df07.drop_duplicates().dropna()
#df08=df08.drop_duplicates().dropna()
df09=df09.drop_duplicates().dropna()

# Compute elapsed time
elapsed_time = time.time() - start_time

# Print elapsed time
print(elapsed_time)
# -

df05.describe()

df05.head()

temp = pd.crosstab(df01['An_Naissance'], df01['Civilite'])
temp.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

# +
# Record start time
start_time = time.time()

print(np.shape(df01),
np.shape(df02),
np.shape(df03),
np.shape(df04),
np.shape(df05),
np.shape(df06),
np.shape(df09))

# Compute elapsed time
elapsed_time = time.time() - start_time

# Print elapsed time
print(elapsed_time)

# Returns (693025, 13) (685774, 7) (247, 4) (618394, 5) (706482, 7) (160108, 9) (1147, 3)

# +
# Record start time
start_time = time.time()

df01_02=pd.merge(df01, df02, on='Idx', how='outer')
df01_03=pd.merge(df01_02, df03, on='Idx', how='outer')
df01_04=pd.merge(df01_02, df04, on='Idx', how='outer')
df01_05=pd.merge(df01_04, df05, on='Idx', how='outer')
df01_06=pd.merge(df01_05, df06, on='Idx', how='outer')
df01_09=pd.merge(df01_06, df09, on='Idx', how='outer')

print(np.shape(df01_09))

# Compute elapsed time
elapsed_time = time.time() - start_time

# Print elapsed time
print(elapsed_time)

# +
# Record start time
start_time = time.time()

df01_09=df01_09.drop_duplicates().dropna()

# Compute elapsed time
elapsed_time = time.time() - start_time

# Print elapsed time
print(elapsed_time)
print(np.shape(df01_09))
# -

df01_09_DateDebut=df01_09[['DateDebut_x', 'DateDebut_y']]
df01_09_DateDebut.head()

plt.plot(df01_09_DateDebut['DateDebut_x'],df01_09_DateDebut['DateDebut_y'])

np.shape(df01_09_DateDebut)

plt.plot(df01_09['DateDebut_x'])




