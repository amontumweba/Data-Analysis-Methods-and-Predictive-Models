# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/gdrive')

automobile_data_processed = pd.read_csv('/gdrive/MyDrive/Files/automobile_data_processed.csv')

plt.figure(figsize=(12, 8))

fg = sns.FacetGrid(automobile_data_processed, col='fuel-type', 
                   height=7, aspect=1)

fg.map(plt.hist, 'price', color='red')

plt.show()

fg = sns.FacetGrid(automobile_data_processed, col='fuel-type', 
                   height=7, aspect=1)

fg.map(sns.scatterplot, 'highway-mpg', 'price', s=100)

fg.add_legend()

fg = sns.FacetGrid(automobile_data_processed, col='fuel-type', 
                   hue='num-of-cylinders', height=7, 
                   aspect=1)

fg.map(sns.scatterplot, 'highway-mpg', 'price', s=100)

fg.add_legend()

# with tips dataset

tips_data = pd.read_csv('/gdrive/MyDrive/Files/tips.csv')

plt.figure(figsize=(15, 10))

fg = sns.FacetGrid(tips_data, col='size', height=7, 
                   aspect=1)

fg.map(plt.hist, 'total_bill', color='green')

plt.show()

group_size_values = np.sort(tips_data['size'].unique())

fg = sns.FacetGrid(tips_data, row='time', 
                   height=6, aspect=1)

fg.map(sns.swarmplot, 'size', 'total_bill', 
       order=group_size_values)

fg.add_legend()

plt.show()

fg = sns.FacetGrid(tips_data, col='time', 
                   height=6, aspect=1)

fg.map(sns.swarmplot, 
       'size', 
       'total_bill', 
       order=group_size_values)

fg.add_legend()

plt.show()

