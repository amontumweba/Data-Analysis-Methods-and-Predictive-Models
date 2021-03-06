# -*- coding: utf-8 -*-
from google.colab import drive
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

drive.mount('/gdrive')

crude_oil_data = pd.read_csv('/gdrive/MyDrive/Files/U.S._crude_oil_production.csv')

crude_oil_data.shape

crude_oil_data.columns

crude_oil_data.columns[(crude_oil_data.sum(axis=0)) == 0]

crude_oil_data.drop(['Arizona', 'Virginia'], inplace=True, axis=1)

crude_oil_data.columns

crude_oil_data['Date'] = pd.to_datetime(crude_oil_data['Month'])

crude_oil_data.drop('Month', inplace=True, axis=1)

crude_oil_data.columns

# renaming names which are long to short and simple names

crude_oil_data = crude_oil_data.rename(columns={'Federal Offshore Gulf of Mexico Crude Oil': 'Mexico', 
                                                'Federal Offshore Pacific Crude Oil': 'Pacific'})

crude_oil_data.columns

crude_oil_data['Year'] = crude_oil_data['Date'].dt.year

crude_oil_data['Year'].sample(10)

crude_oil_data['Month'] = crude_oil_data['Date'].dt.month

crude_oil_data['Month'].sample(10)

crude_oil_data.to_csv('/gdrive/MyDrive/Files/crude_oil_data_processed.csv', index=False)

crude_oil_data.describe()

