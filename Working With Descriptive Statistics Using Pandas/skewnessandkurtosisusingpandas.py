# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive

drive.mount('/gdrive')

height_weight_data = pd.read_csv('/gdrive/MyDrive/Files/500_Person_Gender_Height_Weight_Index.csv')

height_weight_data.drop('Index', inplace=True, axis=1)

height_weight_data[['Height']].plot(kind='kde', title='Height', figsize=(10, 8))

plt.show()

height_weight_data[['Weight']].plot(kind='kde', title='Weight', figsize=(10, 8))

plt.show()

height_weight_data['Height'].skew()

height_weight_data['Weight'].skew()

listOfSeries = [pd.Series(['Male', 400, 300], index=height_weight_data.columns), 
                pd.Series(['Female', 660, 370], index=height_weight_data.columns), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns), 
                pd.Series(['Male', 202, 390], index=height_weight_data.columns), 
                pd.Series(['Female', 770, 210], index=height_weight_data.columns), 
                pd.Series(['Male', 880, 203], index=height_weight_data.columns)]

height_weight_updated = height_weight_data.append(listOfSeries, ignore_index=True)

height_weight_updated[['Height']].plot(kind='hist', bins=100, 
                                       title='Height', figsize=(10, 8))

plt.show()

height_weight_updated[['Weight']].plot(kind='hist', bins=100, 
                                       title='Weight', figsize=(10, 8))

plt.show()

height_weight_updated[['Height']].plot(kind='kde',
                                       title='Height', figsize=(10, 8))

plt.show()

height_weight_updated[['Weight']].plot(kind='kde',
                                       title='Weight', figsize=(10, 8))

plt.show()

height_weight_updated['Height'].skew()

height_weight_updated['Weight'].skew()

# Kurtosis --> A measure of the likelihood of extreme events -a normal distribution has a kurtosis of 3

height_weight_data['Height'].kurtosis() # extreme events are less likely to happen

height_weight_data['Weight'].kurtosis()

height_weight_updated['Height'].kurtosis()

height_weight_updated['Weight'].kurtosis()

