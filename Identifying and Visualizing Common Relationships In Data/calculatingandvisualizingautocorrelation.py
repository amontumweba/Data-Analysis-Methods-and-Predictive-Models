# -*- coding: utf-8 -*-

from google.colab import drive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/gdrive')

bikesharing_data = pd.read_csv('/gdrive/MyDrive/Files/bike_sharing_hourly.csv', index_col=0)

bikesharing_data.columns[(bikesharing_data.sum(axis=0))==0]

bikesharing_data[['temp', 'hum']].describe()

bikesharing_data[['temp', 'hum']].corr()

bikesharing_data['temp'].autocorr(lag=2)

# this is indicates a temperature of 2 hours ago is a strong predictor of temp now

bikesharing_data['temp'].autocorr(lag=12)

bikesharing_data['temp'].autocorr(lag=102)

bikesharing_data['temp'].autocorr(lag=1002)

def autocorr(data):
  lst = [2, 12, 24, 102, 1002]
  autc1 = {}
  for i in lst:
    autc1[i] = bikesharing_data[data].autocorr(lag=i)
  return autc1

autocorr('temp')

autocorr('hum')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.acorr(bikesharing_data['temp'], maxlags=12, color='green')
ax1.title.set_text('Temperature')
ax1.set_xlabel('Lags', fontsize=15)

ax2.acorr(bikesharing_data['hum'], maxlags=12, color='red')
ax2.title.set_text('Humidity')
ax2.set_xlabel('Lags', fontsize=15)

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.acorr(bikesharing_data['temp'], maxlags=24, color='deeppink')
ax1.set_xlabel('Lags', fontsize=15)

ax2.acorr(bikesharing_data['hum'], maxlags=24, color='blue')
ax2.set_xlabel('Lags', fontsize=15)

plt.suptitle('Autocorrelation')

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.acorr(bikesharing_data['temp'], maxlags=48, color='red')
ax1.title.set_text('Temperature')
ax1.set_xlabel('Lags', fontsize=12)

ax2.acorr(bikesharing_data['hum'], maxlags=48, color='black')
ax2.title.set_text('Humidity')
ax2.set_xlabel('Lags', fontsize=12)

plt.show()

