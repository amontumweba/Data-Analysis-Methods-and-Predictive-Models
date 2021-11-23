# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas.util.testing as tm

import statsmodels

from google.colab import drive
from scipy import stats
from statsmodels.stats import stattools
from statsmodels.stats.weightstats import DescrStatsW

drive.mount('/gdrive')

mall_data = pd.read_csv('/gdrive/MyDrive/Files/mall_data_processed.csv')

income_descr = DescrStatsW(mall_data['annual_income'])

age_descr = DescrStatsW(mall_data['age'])

q1_income = income_descr.quantile(0.25)

q3_income = income_descr.quantile(0.75)

q1_income

q3_income

iqr_income = q3_income.loc[0.75] - q1_income.loc[0.25]

iqr_income

stats.iqr(mall_data['annual_income']) # Either the q1 or q3 percentile boundary lies between two data points 
# - default linear interpolation is performed

stats.iqr(mall_data['annual_income'], interpolation='lower')

stats.iqr(mall_data['annual_income'], interpolation='higher')

stats.iqr(mall_data['annual_income'], interpolation='midpoint')

q1_income_np = np.percentile(mall_data['annual_income'], 25)

q1_income_np

q3_income_np = np.percentile(mall_data['annual_income'], 75)

q3_income_np

plt.figure(figsize=(10, 8))

sns.boxplot(x='gender', y='annual_income', hue='gender', data=mall_data, orient='v')

plt.show()

plt.figure(figsize=(10, 8))

sns.boxplot(x='gender', y='spending_score', hue='gender', data=mall_data, orient='v')

plt.show()

plt.figure(figsize=(10, 8))

sns.boxplot(x='above_average_income', y='spending_score', hue='above_average_income', 
            data=mall_data, orient='v')

plt.show()

income_descr.var

age_descr.var

income_descr.std

age_descr.std

stats.describe(mall_data['annual_income'])

stats.describe(mall_data['age'])

