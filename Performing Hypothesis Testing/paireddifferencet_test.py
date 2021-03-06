# -*- coding: utf-8 -*-

!pip install researchpy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from google.colab import drive

from scipy import stats
import researchpy as rp

import pandas.util.testing as tm

drive.mount('/gdrive')

bp_reading = pd.read_csv('/gdrive/MyDrive/Files/blood_pressure.csv')

bp_reading.sample(10)

bp_reading.shape

bp_reading.describe().T

bp_reading[['bp_before', 'bp_after']].boxplot(figsize=(10, 8))
plt.show()

stats.levene(bp_reading['bp_after'], bp_reading['bp_before'])

bp_reading['bp_diff'] = scale(bp_reading['bp_after'] - bp_reading['bp_before'])

bp_reading[['bp_diff']].head()

bp_reading[['bp_diff']].hist(figsize=(10, 8))
plt.show()

plt.figure(figsize=(10, 8))
stats.probplot(bp_reading['bp_diff'], plot=plt)

plt.title('Blood pressure difference Q-Q plot')
plt.show()

stats.shapiro(bp_reading['bp_diff'])

stats.ttest_rel(bp_reading['bp_after'], bp_reading['bp_before'])

rp.ttest(bp_reading['bp_after'], bp_reading['bp_before'], 
         paired=True, equal_variances=False)

