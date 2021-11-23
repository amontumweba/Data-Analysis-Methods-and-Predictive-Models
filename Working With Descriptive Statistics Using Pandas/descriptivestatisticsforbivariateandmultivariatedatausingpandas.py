# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive

drive.mount('/gdrive')

automobile_data = pd.read_csv('/gdrive/MyDrive/Files/auto-mpg.csv')

automobile_data.shape

automobile_data = automobile_data.replace('?', np.nan)
automobile_data = automobile_data.dropna()

automobile_data.shape

automobile_data.drop(['origin', 'car name'], axis=1, inplace=True)

automobile_data['model year'] = '19' + automobile_data['model year'].astype(str)

import datetime

automobile_data['age'] = datetime.datetime.now().year - \
pd.to_numeric(automobile_data['model year'])

automobile_data.drop('model year', inplace=True, axis=1)

automobile_data.dtypes

automobile_data['horsepower'] = pd.to_numeric(automobile_data['horsepower'], errors='coerce')

automobile_data.to_csv('/gdrive/MyDrive/Files/automobile_data_processed.csv', index=False)

automobile_data.plot.scatter(x='displacement', y='mpg', figsize=(10, 8))

plt.show()

fig, ax = plt.subplots()

automobile_data.plot(x='horsepower', y='mpg', kind='scatter', s=60, 
                     c='cylinders', cmap='magma_r', title='Automobile Data', 
                     figsize=(10,8), ax=ax)

plt.show()

automobile_data['acceleration'].cov(automobile_data['mpg'])

automobile_data['acceleration'].corr(automobile_data['mpg'])

automobile_data['horsepower'].cov(automobile_data['mpg'])

automobile_data['horsepower'].corr(automobile_data['mpg'])

automobile_data['horsepower'].cov(automobile_data['displacement'])

automobile_data['horsepower'].corr(automobile_data['displacement'])

automobile_data_cov = automobile_data.cov()

automobile_data_cov

automobile_data_corr = automobile_data.corr()

automobile_data_corr

plt.figure(figsize=(10, 8))

sns.heatmap(automobile_data_corr, annot=True)

plt.show()

