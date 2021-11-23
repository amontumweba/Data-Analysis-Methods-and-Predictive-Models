# -*- coding: utf-8 -*-

! pip install researchpy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import researchpy as rp

from statsmodels.formula.api import ols
from google.colab import drive

drive.mount('/gdrive')

bike_sharing_data = pd.read_csv('/gdrive/MyDrive/Files/bike_sharing_data_daily_processed.csv')

bike_sharing_data.head()

bike_sharing_data.shape

bike_sharing_data['weathersit'].unique()

bike_sharing_data.groupby('weathersit')['cnt'].describe().T

bike_sharing_data.boxplot(column=['cnt'], by='weathersit', figsize=(10, 8))

stats.f_oneway(bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 1], 
               bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 2], 
               bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 3])

# A smallest p-value implies that the weather situation on a particular day does have a significant effect
# on the number of bikes rented

# But what does on-way ANOVA doesn't tell is how do the means of the individual groups compared
# against other groups, in this case we use Tukey's Honest Significance Difference: tese to find which
# specific group's means compared with each other are different

# How to perform Tukey's Honest Significance Differnce test

from statsmodels.stats.multicomp import MultiComparison

mul_com = MultiComparison(bike_sharing_data['cnt'], bike_sharing_data['weathersit'])

mul_result = mul_com.tukeyhsd()

print(mul_result)

