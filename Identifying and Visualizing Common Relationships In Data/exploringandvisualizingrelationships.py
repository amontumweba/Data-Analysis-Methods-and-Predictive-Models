# -*- coding: utf-8 -*-

from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

drive.mount('/gdrive')

crude_oil_data = pd.read_csv('/gdrive/MyDrive/Files/crude_oil_data_processed.csv')

crude_oil_data['Date'] = pd.to_datetime(crude_oil_data['Date'])

crude_oil_data.columns

crude_oil_data.plot(x='Date', y='U.S. Crude Oil ', 
                    figsize=(12, 8), color='brown')

plt.ylabel('Production')
plt.title('US Crude Oil Production')
plt.show()

# To view the summary statistics is the boxplot

crude_oil_data.boxplot('U.S. Crude Oil ', figsize=(12, 8))

plt.ylabel('Production')
plt.title('U.S Crude Oil Production')
plt.show()

crude_oil_data.boxplot('California', figsize=(12, 8))

plt.ylabel('Production in California')
plt.title('California Crude Oil Production')
plt.show()

crude_oil_data[['Alaska', 'California']].boxplot(figsize=(12, 8))
plt.ylabel('Production in Alaska, California')
plt.title('Alaska, California Crude Oil Production')
plt.show()

## The figure below shows that: We have positive outliers in Alaska and Negative outliers in California

crude_oil_data.boxplot(column=['U.S. Crude Oil '], by=['Year'], 
                       figsize=(12, 8))
plt.ylabel('Production')
plt.title('U.S. Crude Oil')
plt.show()

# Year Production

year_data = crude_oil_data.groupby('Year', as_index=False).sum()

colors = ['C0', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

year_data.plot.bar(x='Year', y='U.S. Crude Oil ', 
                   figsize=(12, 8), color=colors, 
                   legend=False)

plt.ylabel('Production')
plt.title('U.S Crude Oil Production')
plt.show()

# Average Crude Oil Production for all of the individual states that exist in out dataset

mean_prod_data = crude_oil_data.mean()[1:-3]

mean_prod_data = mean_prod_data.sort_values(ascending=False)

mean_prod_data

mean_prod_df = pd.DataFrame(mean_prod_data).reset_index()
mean_prod_df.columns = ['State', 'Production']

mean_prod_df.head(10)

# To view the average Oil production across each state is the Bar Plot

plt.figure(figsize=( 12, 8))

plt.bar(mean_prod_df['State'], mean_prod_df['Production'], 
        width=0.85)

plt.title('US Oil Mean-Production June 2008 to June 2018')

plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Production')

plt.show()

# Visualizing Oil Production across each state in form of Probability density Distribution, 
# you can use the  kde curve
# kde stands for Kernel density Estimation
mean_prod_df['Production'].plot.kde(figsize=(12, 8))

plt.title('US Oil Production June 2008 to June 2018')
plt.xlabel('Oil Production')

plt.show()

# Scatter plot

plt.figure(figsize=(12, 8))

plt.scatter(crude_oil_data['Texas'], crude_oil_data['U.S. Crude Oil '], c='g')

plt.xlabel('US Production')
plt.ylabel('Texas Production')

plt.show()

plt.figure(figsize=(12, 8))

plt.scatter(crude_oil_data['California'], crude_oil_data['U.S. Crude Oil '], c='y')

plt.xlabel('US Production')
plt.ylabel('California Production')

plt.show()

plt.figure(figsize=(12, 8))

plt.scatter(crude_oil_data['Michigan'], crude_oil_data['U.S. Crude Oil '], c='g')

plt.xlabel('US Production')
plt.ylabel('Michigan Production')

plt.show()

