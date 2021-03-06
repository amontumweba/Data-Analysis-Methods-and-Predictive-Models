# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

drive.mount('/gdrive')

bikesharing_data = pd.read_csv('/gdrive/MyDrive/Files/bike_sharing_daily.csv', index_col=0)

bikesharing_data['dteday'] = pd.DatetimeIndex(bikesharing_data['dteday'])

plt.figure(figsize=(20, 8))

plt.plot(bikesharing_data['dteday'], bikesharing_data['registered'], 
         color='b', linewidth=2, label='registered')

plt.plot(bikesharing_data['dteday'], bikesharing_data['casual'], 
         color='r', linewidth=2, label='casual')

plt.legend(loc='upper left')

plt.title('Bike Sharing Users')
plt.xlabel('Date')
plt.ylabel('Counts of Bike Rentals')

plt.show()

year_df = bikesharing_data.groupby('yr', as_index=False).mean()

year_df[['yr', 'cnt']]

plt.figure(figsize=(12, 8))

colors = ['b', 'm']

plt.bar(year_df['yr'], year_df['cnt'], width=0.2, 
        color=colors)

plt.xticks([0, 1], ['2011', '2012'])

plt.title('Bike Sharing Daily')
plt.xlabel('year')
plt.ylabel('mean count')

plt.show()

days = bikesharing_data.groupby('workingday', as_index=False).mean()

days[['cnt']]

plt.figure(figsize=(12, 8))

plt.bar(days['workingday'], days['cnt'], 
        width=0.2, color=['red', 'limegreen'])

plt.xticks([0, 1], ['Holiday', 'Working day'])

plt.title('Bike Sharing Daily')
plt.xlabel('Days')
plt.ylabel('Average Counts of Rental Bikes')

plt.show()

# Bike sharing for each month

year_data = bikesharing_data.loc[bikesharing_data['yr'] == 1]

month_df = year_data[['mnth', 'cnt']].groupby('mnth', as_index=False).mean()

month_df['mnth'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 
                                    'Oct', 'Nov', 'Dec'], inplace=True)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'c', 'limegreen', 'deeppink']

plt.figure(figsize=(12, 8))

plt.bar(month_df['mnth'], month_df['cnt'], color=colors)

plt.title('Bike Sharing Daily')
plt.xlabel('Months')
plt.ylabel('Average Counts of Bike Rentals')

plt.show()

# Statistical distribution of Bikes using a box plot


plt.figure(figsize=(12, 8))

plt.boxplot(bikesharing_data['cnt'])

plt.xticks([1], ['Rental Bikes'])
plt.title('Bike Sharing Daily')
plt.ylabel('Total Counts of Rental Bikes')

plt.show()

selected_bike_data = bikesharing_data[['casual', 'registered']]

columns = selected_bike_data.columns

bike_data_array = selected_bike_data.values

colors = ['g', 'm']

plt.figure(figsize=(12, 8))

bp = plt.boxplot(bike_data_array, patch_artist=True, labels=columns)

for i in range(len(bp['boxes'])):
  bp['boxes'][i].set(facecolor=colors[i])

plt.title('Bike Sharing Users')
plt.xlabel('Users')
plt.ylabel('Counts of Bike Rentals')

plt.show()

# Violinplot

plt.figure(figsize=(12, 8))

bp = plt.violinplot(bike_data_array)

plt.xticks([1, 2], columns)

plt.title('Bike Sharing Users')
plt.xlabel('Users')
plt.ylabel('Counts of Bike Rentals')

plt.show()

# Pie chat

season_data = bikesharing_data[['season', 'cnt']]

grouped_data = season_data.groupby('season', as_index=False).mean()

grouped_data

grouped_data.replace([1, 2, 3, 4], ['spring', 'summer', 'fall', 'winter'], inplace=True)

grouped_data

plt.figure(figsize=(12, 8))

plt.pie(grouped_data['cnt'], labels=grouped_data['season'], 
        autopct='%.1f')

plt.suptitle('Percentage count of Bike Rentals by Season')
plt.show()

plt.figure(figsize=(12, 8))

plt.pie(grouped_data['cnt'], labels=grouped_data['season'], autopct='%.1f', 
        wedgeprops= {'linewidth': 4, 'edgecolor': 'white'})

plt.suptitle('Percentage counts of Bike Rentals per Season')

plt.show()

explode_max = (0, 0, 0.2, 0)

explode_min = (0.2, 0, 0, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

wedges, texts, autotexts = ax1.pie(grouped_data['cnt'], labels=grouped_data['season'], 
                                   autopct='%.1f', explode=explode_max)

wedges[2].set(edgecolor='k', linewidth=2)

wedges, texts, autotexts = ax2.pie(grouped_data['cnt'], labels=grouped_data['season'], autopct='%.1f', 
                                   explode=explode_min)

wedges[0].set(edgecolor='k', linewidth=2)

plt.suptitle('Percentage count of Bike Rentals by Season')

plt.show()

