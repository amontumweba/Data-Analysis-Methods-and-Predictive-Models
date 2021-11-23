# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from google.colab import drive

drive.mount('/gdrive')

data = pd.read_csv('/gdrive/MyDrive/Files/weight-height.csv')

data.shape

data.describe().T

sns.countplot(data['Gender'])

plt.show()

height = data['Height']

plt.figure(figsize=(12, 8))

height.plot(kind='hist', title='Hieght Histogram')

plt.show()

plt.figure(figsize=(10, 8))

height.plot(kind='box', title='Height Box-plot')

plt.show()

height.plot(kind='kde', title='Height KDE', figsize=(10, 8))
plt.show()

weight = data['Weight']

plt.figure(figsize=(10, 8))

weight.plot(kind='kde', title='Weight KDE')

plt.show()

# two peaks represent a bi-model representation

plt.figure(figsize=(10, 8))

sns.scatterplot(x='Height', y='Weight', data=data)

plt.show()

plt.figure(figsize=(10, 8))

sns.scatterplot(x='Height', y='Weight', hue='Gender', data=data)

plt.show()

gender_groupby = data.groupby('Gender', as_index=False)

gender_groupby.head()

gender_groupby.describe().T

sns.FacetGrid(data, hue='Gender', height=5).map(sns.histplot, 'Height').add_legend()
plt.show()

sns.FacetGrid(data, hue='Gender', height=5).map(sns.histplot, 'Weight').add_legend()
plt.show()

