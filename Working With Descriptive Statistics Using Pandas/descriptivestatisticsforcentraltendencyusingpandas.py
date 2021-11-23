# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive

drive.mount('/gdrive')

height_weight_data = pd.read_csv('/gdrive/MyDrive/Files/500_Person_Gender_Height_Weight_Index.csv')

height_weight_data.drop('Index', inplace=True, axis=1)

height_weight_data.shape

height_weight_data.isnull().sum()

min_height = height_weight_data['Height'].min()

min_height

max_height = height_weight_data['Height'].max()

max_height

min_weight = height_weight_data['Weight'].min()

min_weight

max_weight = height_weight_data['Weight'].max()

max_weight

range_of_height = max_height - min_height

range_of_height

range_of_weight = max_weight - min_weight

range_of_weight

weight = height_weight_data['Weight']

sorted_weight = weight.sort_values().reset_index(drop=True)

sorted_weight.head()

def mean_value(data):
  num_elements = len(data)
  print('Number of elements: ', num_elements)

  weight_sum = sum(data)
  print('Sum: ', weight_sum)

  return weight_sum/num_elements

def median_value(data):
  num_elements = len(data)

  if (num_elements % 2 == 0):
    return (data[(num_elements / 2) - 1] + data[(num_elements / 2)]) / 2

  else:
    return (data[((num_elements + 1) / 2) - 1])

mean_value(height_weight_data['Weight'])

weight_mean = height_weight_data['Weight'].mean()

weight_mean

median_value(sorted_weight)

weight_median = height_weight_data['Weight'].median()

weight_median

plt.figure(figsize=(10, 8))

height_weight_data['Weight'].hist(bins=30)

plt.axvline(weight_mean, color='r', label='mean')

plt.legend()

plt.show()

plt.figure(figsize=(10, 8))

height_weight_data['Weight'].hist(bins=30)

plt.axvline(weight_median, color='g', label='median')

plt.legend()

plt.show()

listOfSeries = [pd.Series(['Male', 205, 460], index=height_weight_data.columns), 
                pd.Series(['Female', 202, 390], index=height_weight_data.columns), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns), 
                pd.Series(['Male', 202, 390], index=height_weight_data.columns), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns), 
                pd.Series(['Male', 202, 490], index=height_weight_data.columns)]

height_weight_data_updated = height_weight_data.append(listOfSeries, ignore_index=True)

height_weight_data_updated.tail()

updated_weight_mean = height_weight_data_updated['Weight'].mean()

updated_weight_mean, weight_mean

updated_weight_median = height_weight_data_updated['Weight'].median()

updated_weight_median, weight_median

plt.figure(figsize=(10, 8))

height_weight_data_updated['Weight'].hist(bins=100)

plt.axvline(updated_weight_mean, color='r', label='mean')
plt.axvline(updated_weight_median, color='g', label='median')

plt.legend()

plt.show()

# mode

height_counts = {}

for p in height_weight_data['Height']:
  if p not in height_counts:
    height_counts[p] = 1
  else:
    height_counts[p] += 1

plt.figure(figsize=(25, 10))

x_range = range(len(height_counts))

plt.bar(x_range, list(height_counts.values()), align='center')
plt.xticks(x_range, list(height_counts.keys()))

plt.xlabel('Height')
plt.ylabel('Count')

plt.show()

count = 0
size = 0

for s, c in height_counts.items():
  if count < c:
    counts = c
    size = s

print('Size: ', size, '\nFrequency: ', count)

mode_height = height_weight_data['Height'].mode()

mode_height

mode_weight = height_weight_data['Weight'].mode()

mode_weight

height_mean = height_weight_data['Height'].mean()

height_median = height_weight_data['Height'].median()

height_mode = height_weight_data['Height'].mode().values[0]

plt.figure(figsize=(12, 8))

height_weight_data['Height'].hist()

plt.axvline(height_mean, color='r', label='mean')
plt.axvline(height_median, color='g', label='median')
plt.axvline(height_mode, color='y', label='mode')

plt.legend()

plt.show()

