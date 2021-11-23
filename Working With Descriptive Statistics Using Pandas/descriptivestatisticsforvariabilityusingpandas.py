# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive

drive.mount('/gdrive')

height_weight_data = pd.read_csv('/gdrive/MyDrive/Files/500_Person_Gender_Height_Weight_Index.csv')

height_weight_data.drop('Index', inplace=True, axis=1)

height_weight_data.shape

num_records = height_weight_data.shape[0]

height_data = height_weight_data[['Height']].copy()

weight_data = height_weight_data[['Weight']].copy()

counts = [1] * num_records

height_data['counts_height'] = counts
weight_data['counts_weight'] = counts

weight_data = weight_data.sort_values('Weight')

weight_data.tail()

height_data = height_data.sort_values('Height')

height_data = height_data.groupby('Height', as_index=False).count()

weight_data = weight_data.groupby('Weight', as_index=False).count()

height_data['cumcounts_height'] = height_data['counts_height'].cumsum()

weight_data['cumcounts_weight'] = weight_data['counts_weight'].cumsum()

# Quantile

ql_height = height_weight_data['Height'].quantile(.25)
q3_height = height_weight_data['Height'].quantile(.75)
iqr_height = q3_height - ql_height

print('25th quantile: ', ql_height)
print('75th quantile: ', q3_height)
print('Interquantile range: ', iqr_height)

plt.figure(figsize=(12, 8))

height_weight_data['Height'].hist(bins=30)

plt.axvline(ql_height, color='r', label='Q1')
plt.axvline(q3_height, color='g', label='Q3')

plt.legend()

plt.show()

plt.figure(figsize=(12, 8))

plt.scatter(height_weight_data['Weight'], height_weight_data['Height'], s=100)

plt.axvline(height_weight_data['Weight'].quantile(.25), color='r', label='Q1 Weight')
plt.axvline(height_weight_data['Weight'].quantile(.75), color='g', label='Q2 Weight')


plt.axhline(height_weight_data['Height'].quantile(.25), color='y', label='Q1 Height')
plt.axhline(height_weight_data['Height'].quantile(.75), color='m', label='Q2 Hieght')


plt.legend()

plt.show()

plt.figure(figsize=(12, 8))

plt.bar(height_data['Height'], height_data['cumcounts_height'])

plt.axvline(height_weight_data['Height'].quantile(.25), color='y', label='25%')
plt.axvline(height_weight_data['Height'].quantile(.50), color='m', label='50%')
plt.axvline(height_weight_data['Height'].quantile(.75), color='r', label='75%')

plt.legend()

plt.show()

# Variance

def variance(data):
  diffs = 0
  avg = sum(data) / len(data)

  for n in data:
    diffs += (n - avg) ** 2

  return (diffs/(len(data)-1))

variance(height_weight_data['Height'])

variance(height_weight_data['Weight'])

height_weight_data['Height'].var()

height_weight_data['Weight'].var()

std_height = (variance(height_weight_data['Height'])) ** 0.5

std_height

std_weight = (variance(height_weight_data['Weight'])) ** 0.5

std_weight

height_weight_data['Height'].std()

height_weight_data['Weight'].std()

weight_mean = height_weight_data['Weight'].mean()

weight_std = height_weight_data['Weight'].std()

listOfSeries = [pd.Series(['Male', 40, 30], index=height_weight_data.columns), 
                pd.Series(['Female', 66, 37], index=height_weight_data.columns), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns), 
                pd.Series(['Male', 202, 390], index=height_weight_data.columns), 
                pd.Series(['Female', 77, 210], index=height_weight_data.columns), 
                pd.Series(['Male', 88, 203], index=height_weight_data.columns)]

height_weight_updated = height_weight_data.append(listOfSeries, ignore_index=True)

plt.figure(figsize=(12, 8))

height_weight_updated['Weight'].hist(bins=100)

plt.show()

plt.figure(figsize=(12, 8))

height_weight_updated['Height'].hist(bins=100)

plt.show()

height_weight_updated['Height'].quantile(.25)

height_weight_updated['Height'].quantile(.75)

