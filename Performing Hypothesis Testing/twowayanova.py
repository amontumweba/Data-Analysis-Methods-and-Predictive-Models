# -*- coding: utf-8 -*-

! pip install researchpy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from google.colab import drive
import pandas.util.testing as tm

drive.mount('/gdrive')

bike_sharing_data = pd.read_csv('/gdrive/MyDrive/Files/bike_sharing_data_daily_processed.csv')

bike_sharing_data.head()

# we are going to work with two categorical variables

bike_sharing_data['weathersit'].unique()

bike_sharing_data['season'].unique()

rp.summary_cont(bike_sharing_data.groupby(['weathersit']))['cnt']

bike_sharing_data.boxplot(column=['cnt'], by='weathersit', figsize=(10, 8))

rp.summary_cont(bike_sharing_data.groupby(['season']))['cnt']

bike_sharing_data.boxplot(column=['cnt'], by='season', figsize=(10, 8))
plt.show()

X = sm.add_constant(bike_sharing_data)

model = ols('cnt ~ C(weathersit)', X).fit()

print(model.summary())

model1 = ols('cnt ~ C(weathersit)', bike_sharing_data).fit()

print(model1.summary())

model = ols('cnt ~ C(season)', bike_sharing_data).fit()
print(model.summary())

model2 = ols('cnt ~ C(weathersit) + C(season)', bike_sharing_data).fit() # considering both weathersit and season with interaction
print(model2.summary())

sm.stats.anova_lm(model2)

model3 = ols('cnt ~ C(weathersit) * C(season)', bike_sharing_data).fit()
# weathersit and season interaction
print(model3.summary())

sm.stats.anova_lm(model3)

