# -*- coding: utf-8 -*-

!pip install researchpy

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from google.colab import drive
import pandas.util.testing as tm

import researchpy as rp
from scipy import stats

drive.mount('/gdrive')

bike_sharing_data = pd.read_csv('/gdrive/MyDrive/Files/bike_sharing_daily.csv')

bike_sharing_data.shape

# Using t-test to determine whether the effect of working day and temperature on the number of 
# bikes that were rented is significant

bike_sharing_data = bike_sharing_data[['season', 'mnth', 'holiday', 
                                       'workingday', 'weathersit', 
                                       'temp', 'cnt']]

bike_sharing_data.to_csv('/gdrive/MyDrive/Files/bike_sharing_data_daily_processed.csv', index=False)

bike_sharing_data.head()

bike_sharing_data['season'].unique()
# 1 --> Spring
# 2 --> Summer
# 3 --> Fall
# 4 --> Winter

bike_sharing_data['workingday'].unique()

bike_sharing_data['holiday'].unique()

bike_sharing_data['weathersit'].unique()

# 1 --> clear, few clouds
# 2 --> Misty, cloudy
# 3 --> Light snow, rain, thunderstorm

bike_sharing_data['temp'].describe()

bike_sharing_data.shape

# We will use the t-test to compare the differences in means between 2 samples of population
# and see whether the differences are actually significant

# The first thing we will use t-test for is to see whether the average number of bikes rented
# on a working day vs on a non-working day is different and whether this difference is significant

bike_sharing_data.groupby('workingday')['cnt'].describe()

bike_sharing_data.boxplot(column=['cnt'], by='workingday', figsize=(10, 8))
plt.show()

sample_01 = bike_sharing_data[(bike_sharing_data['workingday'] == 1)]
sample_02 = bike_sharing_data[(bike_sharing_data['workingday'] == 0)]

sample_01.shape, sample_02.shape

sample_01 = sample_01.sample(231)

sample_01.shape, sample_02.shape

# Checking whether the assumptions we make for the t-test are satsified

# 1 ==> Levene's test: To check whether the variance of the two groups are the same, like
# the t-test but for variance rather than mean

stats.levene(sample_01['cnt'], sample_02['cnt'])

# 2 ==> The distribution of the residuals between the two groups should follow the normal distributions

diff = scale(np.array(sample_01['cnt']) - np.array(sample_02['cnt'], dtype=np.float)) # calculating the residuals between two groups

plt.hist(diff) # plotting a histogram
plt.show()

plt.figure(figsize=(10, 8))

stats.probplot(diff, plot=plt, dist='norm') # Generates a probability plot of sample data against the quantiles
# of a theoretical distribution

plt.show()

# Another way to check the normal distribution is to use Shapiro-Wilk test for normality - if the 
# test is not significant then population is normally distributed

stats.shapiro(diff)

stats.ttest_ind(sample_01['cnt'], sample_02['cnt']) # since p-value is greater than 0.05, we accept the null hypothesis
# We can then say that, whether it is a working or not has no effects on the bikes shared

descriptives, results = rp.ttest(sample_01['cnt'], sample_02['cnt'])

descriptives

print(results)

# Performing Welch's t-test

bike_sharing_data.head()

bike_sharing_data[['temp']].boxplot(figsize=(10, 6))
plt.show()

bike_sharing_data['temp_category'] = bike_sharing_data['temp'] > bike_sharing_data['temp'].mean()

bike_sharing_data.groupby('temp_category')['cnt'].describe()

sample_001 = bike_sharing_data[(bike_sharing_data['temp_category'] == True)]
sample_002 = bike_sharing_data[(bike_sharing_data['temp_category'] == False)]

sample_001.shape, sample_002.shape

sample_001 = sample_001.sample(364)

sample_001.shape, sample_002.shape

stats.levene(sample_001['cnt'], sample_002['cnt']) # A signficant levene test implies that we have to reject the null hypothesis and accept the alternative hypothesis

diff1 = scale(np.array(sample_001['cnt']) - np.array(sample_002['cnt']))
plt.hist(diff1)
plt.show()

# Using the probabiliy plot

plt.figure(figsize=(10, 8))
stats.probplot(diff1, plot=plt)
plt.show()

stats.shapiro(diff1)

stats.ttest_ind(sample_001['cnt'], sample_002['cnt'])

descriptives1, results1 = rp.ttest(sample_001['cnt'], sample_002['cnt'], equal_variances=False)

descriptives1

results1

