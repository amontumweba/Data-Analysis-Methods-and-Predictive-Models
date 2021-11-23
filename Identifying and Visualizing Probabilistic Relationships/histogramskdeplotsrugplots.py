# -*- coding: utf-8 -*-
from google.colab import drive
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/gdrive')

automobile_data = pd.read_csv('/gdrive/MyDrive/Files/Automobile_data_new.csv', na_values='?')

automobile_data.shape

automobile_data.isnull().sum()

automobile_data.dropna(inplace=True)

automobile_data.shape

automobile_data.to_csv('/gdrive/MyDrive/Files/automobile_data_processed.csv', index=False)

plt.figure(figsize=(12, 8))

sns.distplot(automobile_data['price'], color='red')

plt.title('Automobile Data')

plt.show()

# with bins

plt.figure(figsize=(12, 8))

sns.distplot(automobile_data['price'], bins=20, color='red')

plt.title('Automobile Data')

plt.show()

plt.figure(figsize=(12, 8))

sns.distplot(automobile_data['price'], hist=False, color='blue')

plt.title('Automobile Data')

plt.show()

plt.figure(figsize=(12, 8))

sns.distplot(automobile_data['price'], hist=False, rug=True, color='blue')

plt.title('Automobile Data')

plt.show()

plt.figure(figsize=(12, 8))

sns.rugplot(automobile_data['price'], height= 0.5, color='blue')

plt.title('Automobile Data')

plt.show()

plt.figure(figsize=(12, 8))

sns.kdeplot(automobile_data['price'], shade=True, color='blue')

plt.title('Automobile Data')

plt.show()

# Visualizing Bivariate distribution

plt.figure(figsize=(12, 8))

sns.scatterplot(x='horsepower', y='price', data=automobile_data, s=120)

plt.title('Automobile Data')

plt.show()

plt.figure(figsize=(12, 8))

sns.scatterplot(x='horsepower', y='price', data=automobile_data, hue='num-of-cylinders', s=120)

plt.title('Automobile Data')

plt.show()

sns.regplot(x='horsepower', y='price', data=automobile_data)

plt.show()

sns.regplot(x='highway-mpg', y='price', data=automobile_data)

plt.show()

# Viewing univariate and Bivariate in the same graph

sns.jointplot(x='horsepower', y='price', data=automobile_data)

plt.show()

sns.jointplot(x='horsepower', y='price', data=automobile_data, kind='reg')

plt.show()

# Kde curve

sns.jointplot(x='horsepower', y='price',shade=True, data=automobile_data, kind='kde')

plt.show()

# hex plot

sns.jointplot(x='horsepower', y='price', data=automobile_data, kind='hex')

plt.show()

f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(automobile_data['horsepower'], automobile_data['price'], ax=ax)

sns.rugplot(automobile_data['horsepower'], color='limegreen', ax=ax)
sns.rugplot(automobile_data['price'], color='red', vertical=True, ax=ax)

plt.title('Automobile Data')

plt.show()

