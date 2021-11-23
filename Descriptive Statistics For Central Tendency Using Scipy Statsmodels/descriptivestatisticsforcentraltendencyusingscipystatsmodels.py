# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy
import statsmodels
import seaborn as sns
import matplotlib.pyplot as plt
import pandas.util.testing as tm

from google.colab import drive
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

drive.mount('/gdrive')

salary_data = pd.read_csv('/gdrive/MyDrive/Files/Salary_Data.csv')

salary_data.shape

salary_data.isnull().sum()

min_exp = np.min(salary_data['YearsExperience'])

min_exp

max_exp = np.max(salary_data['YearsExperience'])

max_exp

min_salary = np.min(salary_data['Salary'])

min_salary

max_salary = np.max(salary_data['Salary'])

max_salary

range_of_exp = np.ptp(salary_data['YearsExperience'])

range_of_exp

range_of_salary = np.ptp(salary_data['Salary'])

range_of_salary

salary = salary_data['Salary']

sorted_salary = salary.sort_values().reset_index(drop=True)

sorted_salary.head()

salary_mean = scipy.mean(salary_data['Salary'])

salary_mean

exp_stats = DescrStatsW(salary_data['YearsExperience'])

exp_stats.mean

salary_median = scipy.median(sorted_salary)

salary_median

salary_median_unSorted = scipy.median(salary_data['Salary'])

salary_median_unSorted

exp_stats.quantile(0.5)

plt.figure(figsize=(12, 8))

sns.distplot(salary_data['Salary'])

plt.show()

plt.figure(figsize=(12, 8))

sns.distplot(salary_data['YearsExperience'])

plt.show()

plt.figure(figsize=(10, 8))

sns.distplot(salary_data['Salary'])

plt.axvline(salary_mean, color='r', label='mean')
plt.axvline(salary_median, color='b', label='median')

plt.legend()

plt.show()

listOfSeries = [pd.Series([20, 250000], index=salary_data.columns), 
                pd.Series([25, 270000], index=salary_data.columns), 
                pd.Series([30, 320000], index=salary_data.columns)]

salary_updated = salary_data.append(listOfSeries, ignore_index=True)

salary_updated.tail()

salary_updated_mean = scipy.mean(salary_updated['Salary'])

salary_updated_mean

salary_updated_median = scipy.median(salary_updated['Salary'])

salary_updated_median

plt.figure(figsize=(10, 8))

sns.distplot(salary_updated['Salary'])

plt.axvline(salary_updated_mean, color='r', label='mean')
plt.axvline(salary_updated_median, color='b', label='median')

plt.legend()

plt.show()

plt.figure(figsize=(10, 8))

sns.distplot(salary_data['Salary'], hist_kws={'alpha':0.2}, color='grey')
sns.distplot(salary_updated['Salary'], hist_kws={'alpha': 0.8}, color='green')

plt.axvline(salary_mean, color='grey', label='mean')
plt.axvline(salary_updated_mean, color='green', label='median')

plt.legend()

plt.show()

plt.figure(figsize=(10, 8))

sns.distplot(salary_data['Salary'], hist_kws={'alpha':0.2}, color='grey')
sns.distplot(salary_updated['Salary'], hist_kws={'alpha': 0.8}, color='green')

plt.axvline(salary_median, color='grey', label='mean')
plt.axvline(salary_updated_median, color='green', label='median')

plt.legend()

plt.show()

stats.mode(salary_data['YearsExperience'])

salary_data['YearsExperience'].mode()

stats.mode(salary_data['Salary'])

salary_data['Salary'].mode().value_counts().sum()

plt.figure(figsize=(10, 8))

sns.countplot(salary_data['YearsExperience'])

plt.show()

plt.figure(figsize=(10, 8))

sns.countplot(salary_data['Salary'])

plt.xticks(rotation=90)

plt.show()

