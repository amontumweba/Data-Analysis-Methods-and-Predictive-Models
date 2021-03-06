# -*- coding: utf-8 -*-

from google.colab import drive
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

drive.mount('/gdrive')

tips_data = pd.read_csv('/gdrive/MyDrive/Files/tips.csv')

tips_data.shape

tips_data.describe()

tips_data.groupby('time').mean()

plt.figure(figsize=(12, 8))

sns.barplot(x='time', y='total_bill', data=tips_data)

plt.title('Bill and Tips')
plt.xticks(rotation=90)

plt.show()

mean = tips_data['tip'].mean()

mean

tips_data['above average'] = (tips_data['tip'] - mean) > 0

tips_data[['tip', 'above average']].head()

plt.figure(figsize=(12, 8))

sns.countplot('time', hue='above average', data=tips_data, 
              order = tips_data['time'].value_counts().index)

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.countplot('time', hue='sex', data=tips_data, 
              order = tips_data['time'].value_counts().index)

plt.title('Bill and Tips')

plt.show()

# how total bill is distributed based on Gender

plt.figure(figsize=(12, 8))

sns.boxplot(x='sex', y='total_bill', data=tips_data, 
            palette='nipy_spectral')

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.boxplot(x='size', y='total_bill', data=tips_data, 
            palette='nipy_spectral')

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.catplot(x='size', y='total_bill', hue='time', data=tips_data, 
            kind='bar', height=6, aspect=2, palette='CMRmap')

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.stripplot(x='size', y='tip', data=tips_data)

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.swarmplot(x='size', y='tip', data=tips_data)

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.swarmplot(x='size', y='total_bill', hue='above average', 
              data=tips_data)

plt.title('Bill and Tips')

plt.show()

plt.figure(figsize=(12, 8))

sns.swarmplot(x='size', y='total_bill', hue='above average', 
              data=tips_data, dodge=True)

plt.title('Bill and Tips')

plt.show()

