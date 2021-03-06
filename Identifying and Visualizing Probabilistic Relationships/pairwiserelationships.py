# -*- coding: utf-8 -*-

from google.colab import drive
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

drive.mount('/gdrive')

automobile_data_processed = pd.read_csv('/gdrive/MyDrive/Files/automobile_data_processed.csv')

automobile_subset = automobile_data_processed[['horsepower', 'city-mpg', 
                                               'highway-mpg', 'price']]

sns.pairplot(automobile_subset)

plt.show()

sns.pairplot(automobile_data_processed, 
             vars=['price', 'horsepower', 'highway-mpg'])

plt.show()

sns.pairplot(automobile_data_processed, 
             vars=['price', 'horsepower', 'highway-mpg'], hue='fuel-type')

plt.show()

g = sns.PairGrid(automobile_data_processed, 
                 vars=['price', 'horsepower', 'highway-mpg'], 
                 hue='fuel-type')

g.map(plt.scatter)

g.add_legend()

plt.show()

g = sns.PairGrid(automobile_data_processed, 
                 vars=['price', 'horsepower', 'highway-mpg'], 
                 hue='fuel-type')

g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)

g.add_legend()

plt.show()

g = sns.PairGrid(automobile_data_processed, 
                 vars=['price', 'horsepower', 'highway-mpg'], 
                 hue='fuel-type')

g.map_lower(plt.scatter)
g.map_diag(sns.kdeplot, lw=3, legend=False)
g.map_upper(sns.regplot)

g.add_legend()

plt.show()

automobile_corr = automobile_data_processed[['engine-size', 'horsepower', 
                                             'peak-rpm', 'city-mpg', 'highway-mpg', 'price']].corr()

automobile_corr

plt.figure(figsize=(12, 8))

sns.heatmap(automobile_corr, vmax=.8, square=True, 
            annot=True, fmt='.2f', cmap='viridis')

plt.title('Automobile Data')

plt.show()

