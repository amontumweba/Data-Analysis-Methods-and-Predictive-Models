# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive

drive.mount('/gdrive')

automobile_data = pd.read_csv('/gdrive/MyDrive/Files/automobile_data_processed.csv')

automobile_data['mpg'].plot.kde(figsize=(10, 8))

plt.xlabel('mpg')

plt.title('Density plot for MPG')
plt.show()

plt.figure(figsize=(10, 8))

sns.boxplot(x='cylinders', y='mpg', data=automobile_data)

plt.show()

plt.figure(figsize=(10, 8))

sns.violinplot(x='cylinders', y='mpg',data=automobile_data, inner=None)
sns.swarmplot(x='cylinders', y='mpg', data=automobile_data, color='w')

plt.show()

cylinders_stats = automobile_data.groupby(['cylinders'])['mpg'].agg(['mean', 'count', 'std'])

cylinders_stats

ci95_high = []

ci95_low = []

for i in cylinders_stats.index:

  mean, count, std = cylinders_stats.loc[i]

  ci95_high.append(mean + 1.96 * (std/math.sqrt(count)))
  ci95_low.append(mean - 1.96 * (std/math.sqrt(count)))

cylinders_stats['ci95_HIGH'] = ci95_high
cylinders_stats['ci95_LOW'] = ci95_low

cylinders_stats

cylinders = 4

cylinders4_df = automobile_data.loc[automobile_data['cylinders'] == cylinders]

plt.figure(figsize=(10, 8))

sns.distplot(cylinders4_df['mpg'], rug=True, kde=True, hist=False)

plt.stem([cylinders_stats.loc[cylinders]['mean']], [0.07], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean', use_line_collection=True)

plt.stem([cylinders_stats.loc[cylinders]['ci95_LOW']], [0.07], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI HIGH', use_line_collection=True)

plt.stem([cylinders_stats.loc[cylinders]['ci95_HIGH']], [0.07], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI LOW', use_line_collection=True)

plt.xlabel('mpg')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))

sns.kdeplot(cylinders4_df['mpg'])

plt.stem([cylinders_stats.loc[cylinders]['mean']], [0.07], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean', use_line_collection=True)

plt.stem([cylinders_stats.loc[cylinders]['ci95_LOW']], [0.07], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI HIGH', use_line_collection=True)

plt.stem([cylinders_stats.loc[cylinders]['ci95_HIGH']], [0.07], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI LOW', use_line_collection=True)

plt.xlabel('mpg')
plt.legend()
plt.show()

cylinders6 = 6

cylinders6_df = automobile_data.loc[automobile_data['cylinders'] == cylinders6]



plt.figure(figsize=(10, 8))

sns.kdeplot(cylinders6_df['mpg'])

plt.stem([cylinders_stats.loc[cylinders6]['mean']], [0.16], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean', use_line_collection=True)

plt.stem([cylinders_stats.loc[cylinders6]['ci95_LOW']], [0.16], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI HIGH', use_line_collection=True)

plt.stem([cylinders_stats.loc[cylinders6]['ci95_HIGH']], [0.16], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI LOW', use_line_collection=True)

plt.xlabel('mpg')
plt.legend()
plt.show()

