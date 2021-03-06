# -*- coding: utf-8 -*-

from google.colab import drive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

drive.mount('/gdrive')

bikesharing_data = pd.read_csv('/gdrive/MyDrive/Files/bike_sharing_daily.csv', index_col=0)

bikesharing_data.columns[(bikesharing_data.sum(axis=0)) == 0]

bikesharing_data.shape

bikesharing_data.columns

bikesharing_data['dteday'] = pd.DatetimeIndex(bikesharing_data['dteday'])

# Correlation coefficient

np.corrcoef(bikesharing_data['temp'], bikesharing_data['cnt'])

# Second way

bikesharing_data['temp'].corr(bikesharing_data['cnt'])

pearsonr(bikesharing_data['temp'], bikesharing_data['cnt'])

spearmanr(bikesharing_data['temp'], bikesharing_data['cnt'])

# Visualizing using a scatter plot

plt.figure(figsize=(12, 8))

plt.scatter(bikesharing_data['temp'], bikesharing_data['cnt'], color='m')

plt.title('Bike Sharing Daily')

plt.xlabel('Temperature')
plt.ylabel('Count')

plt.show()

np.corrcoef(bikesharing_data['workingday'], bikesharing_data['registered'])

bikesharing_data['workingday'].corr(bikesharing_data['registered'])

pearsonr(bikesharing_data['workingday'], bikesharing_data['registered'])

spearmanr(bikesharing_data['workingday'], bikesharing_data['registered'])

bikesharing_data.groupby(['workingday']).max()['registered']

ax = plt.subplot()

bikesharing_data.groupby('workingday').max()['registered'].plot(kind='bar', 
                                                                figsize=(12, 8), 
                                                                ax=ax, color=['r', 'c'])

plt.title('Registered Users')
plt.ylabel('Count of Registered Users')

plt.show()

ax = plt.subplot()

bikesharing_data.groupby('workingday').max()['casual'].plot(kind='bar', 
                                                         figsize=(12, 8), ax = ax, 
                                                         color=['b', 'y'])

plt.title('Casual Users')
plt.ylabel('Count of Casual Users')

plt.show()

np.corrcoef(bikesharing_data['windspeed'], bikesharing_data['cnt'])

bikesharing_data['windspeed'].corr(bikesharing_data['cnt'])

pearsonr(bikesharing_data['windspeed'], bikesharing_data['cnt'])

spearmanr(bikesharing_data['windspeed'], bikesharing_data['cnt'])

plt.figure(figsize=(12, 8))

plt.scatter(bikesharing_data['windspeed'], 
            bikesharing_data['cnt'], color='limegreen')


plt.title('Bike Sharing Daily')
plt.ylabel('Count')
plt.xlabel('Windspeed')

plt.show()

bikesharing_data.corr()

plt.figure(figsize=(12, 8))

plt.matshow(bikesharing_data.corr(), 
            fignum=False, aspect='equal')

columns = len(bikesharing_data.columns)

plt.xticks(range(columns), bikesharing_data.columns)
plt.yticks(range(columns), bikesharing_data.columns)

plt.colorbar()
plt.xticks(rotation=90)
plt.title('Correlations', y=1.2)

plt.show()

from yellowbrick.target import FeatureCorrelation

target = bikesharing_data['cnt']

features = bikesharing_data.drop(['casual', 'registered', 'cnt', 'dteday'], axis=1)

feature_names = list(features.columns)

feature_names

visualizer = FeatureCorrelation(labels=feature_names)

visualizer.fit(features, target)

visualizer.poof()

