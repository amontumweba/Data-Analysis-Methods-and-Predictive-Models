# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from google.colab import drive
import seaborn as sns
from scipy import stats

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# ! pip install ipywidgets

from ipywidgets import interact, interactive, fixed, interact_manual

drive.mount('/gdrive')

url = '/gdrive/MyDrive/Files/imports-85.data'

# Setting headers to numbers
df = pd.read_csv(url, header=None)

# Renaming the headers
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-stype',
           'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
           'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 
           'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

df.columns=headers

path = '/gdrive/MyDrive/Files/automobile.csv'
df.to_csv(path, index=False)

# Checking for data types of each field in df
FieldsDataTypes = df.dtypes

describe = df.describe() # or df.describe(include='all') to put into consideration none numerical fields as well

definition = """
Unique: A number of distinct objects in a column
top: Most frequently occuring object
freq: The number of times the top object appears in a column
"""
definition.splitlines()

# Data Pre-processing or Data cleaning or data wranggling
# 1. Dealing with missing values

df['price'] = df['price'].replace('?', '0')

df['price'] = df['price'].astype('int')

df['price'] = df['price'].dropna()

df = df[df['price'] != 0]

df['normalized-losses'] = df['normalized-losses'].replace('?', '0')

df['normalized-losses'] = df['normalized-losses'].astype('int')

mean = df['normalized-losses'].mean()

df['normalized-losses'] = df['normalized-losses'].replace(0, mean)

# Normalization

df['length'] = df['length']/df['length'].max()

df['width'] = (df['width'] - df['width'].min())/(df['width'].max() - df['width'].min())

df['height'] = (df['height'] - df['height'].mean())/df['height'].std()

# Binning

bins = np.linspace(min(df['price']), max(df['price']), 4)

group_names = ['Low', 'Medium', 'High']

df['price_binned'] = pd.cut(df['price'], bins, labels=group_names, include_lowest=True)

plt.hist(df['price'], bins=3)
plt.title('price bins')
plt.ylabel('count')
plt.xlabel('price')
plt.show()

drive_wheels_counts = pd.DataFrame(df['drive-wheels'].value_counts())

drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'})
drive_wheels_counts.index.name='drive-wheels'
drive_wheels_counts

# Box plot

sns.boxplot(x= 'drive-wheels', y= 'price', data=df)
plt.show()

# Scatter plot

plt.scatter(df['engine-size'], df['price'])

plt.title('Scatterplot of Engine Size vs Price')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.show()

# Group by

df_test = df[['drive-wheels', 'body-stype', 'price']]
 df_grp = df_test.groupby(['drive-wheels', 'body-stype'], as_index=False).mean()

df_grp

df_pivot = df_grp.pivot(index='drive-wheels', columns='body-stype')
df_pivot

# Heat Map

plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# ANAVO --> Analysis of variance
# It can be used to
# 1. find the correlation between different groups of a categorical variable

# ANOVA returns 2 values
# 1. F-test: variation between sample group means divided by variation within sample group
# 2. p-value: confidence degree

df_anova = df[['make', 'price']]

grouped_anova = df_anova.groupby(['make'], as_index=False)

grouped_anova

anova_result_1 = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('subaru')['price'])

f'F-test: {anova_result_1[0]} and p-value: {anova_result_1[1]}'

anova_results_2 = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('jaguar')['price'])

f'F-test: {anova_results_2[0]} and p-value: {anova_results_2[1]}'

anova_results_2[1]

def anova_results(make1, make2, price):
  df_anova = df[['make', 'price']]
  grouped_anova = df_anova.groupby(['make'], as_index=False)
  anova_result = stats.f_oneway(grouped_anova.get_group(make1)[price], grouped_anova.get_group(make2)[price])
  result = f'F-test: {anova_result[0]} and p-value: {anova_result[1]}'
  return result

anova_results('honda', 'subaru', 'price')

anova_results('honda', 'jaguar', 'price')

# Correlation

def regplot(x, y, z):
  sns.regplot(x=x, y=y, data=z)
  plt.ylim(0,)
  return regplot
plt.show()

regplot('engine-size', 'price', df)
plt.show()

regplot('highway-mpg', 'price', df)
plt.show()

df['peak-rpm'] = df['peak-rpm'].replace('?', '0').astype('int')

sns.regplot('peak-rpm', 'price', df)
plt.show()

make = df[['engine-size', 'highway-mpg', 'peak-rpm']]
p = df['price']
a = 2
b = 2
c = 1

fig = plt.figure(figsize=(14, 8))
for i in make:
  plt.subplot(a, b, c)
  plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.25)
  plt.title(f'regplot of {i}')
  sns.regplot(x=make[i], y=p, data=make)
  c+=1

make = df[['engine-size', 'highway-mpg', 'peak-rpm', 'curb-weight']]
p = df['price']
def regplot1(x, y, z):
  a = 2
  b = 2
  c = 1

  fig = plt.figure(figsize=(14, 8))
  for i in make:
    plt.subplot(a, b, c)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.25)
    plt.title(f'regplot of {i}')
    sns.regplot(x=x[i], y=y, data=z)
    c+=1
  
  return regplot1

regplot1(make, p, make)
plt.show()

make = df[['engine-size', 'highway-mpg', 'peak-rpm']]

# Correlation Statistical methods

# 1. Pearson correlation

# Correlation Coefficient
# a. Close to +1: Large Positive relationship
# b. Close to -1: Large Negative relatioship
# c. Close to 0: No relationship


# P-value
# 1. p-value < 0.001: Strong certainty in the result
# 2. p-value < 0.05 : Moderate certainty in the result
# 3. p-value < 0.1: Weak certainty in the result
# 4. p-value > 0.1: No certainty in the result

# Strong Correlation:
# 1. Correlation coefficient close to 1 or -1
# 2. P-value less than 0.001

df['horsepower'] = df['horsepower'].replace('?', '0').astype('int')

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
f'cofficient is: {pearson_coef} and p-value is: {p_value}'

# Correlation between horsePower and Price
def hcorrelation(c, p):
  pearson_coef, p_value = stats.pearsonr(c, p)
  result = f'cofficient is: {pearson_coef} and p-value is: {p_value}'
  return result

hcorrelation(df['horsepower'], df['price'])

# Model developement
from sklearn.linear_model import LinearRegression

# Simple Linear Regression
lm = LinearRegression()

X = df[['highway-mpg']]
Y = df[['price']]

lm.fit(X, Y)

Yhat = lm.predict(Y)

f'Intercept is: {lm.intercept_} and Coefficient is: {lm.coef_}'

# Multiple Linear Regression (MLR)
 Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(Z, Y)

Yhat1 = lm.predict(Z)

f'Intercept is: {lm.intercept_} and Coefficients are: {lm.coef_}'

X = df[['highway-mpg']]
Y = df[['price']]
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

def regression(X, Y):
  lm = LinearRegression()
  lm.fit(X, Y)
  Yhat1 = lm.predict(X)
  Result = f'Intercept is: {lm.intercept_} and Coefficients is/are: {lm.coef_}'
  return Result

regression(X, Y)

regression(Z, Y)

# Residual Plot

x = df['highway-mpg']
y = df['price']
def residplot(x, y):
  sns.residplot(x, y)
  plt.show()
  return residplot

residplot(x, y)
plt.show()

def distplot(y_actual, y_predicted):
  ax1 = sns.distplot(y_actual, hist=False, color='r', label='Actual Value')
  sns.distplot(y_predicted, hist=False, color='b', label='Fitted Values', ax=ax1)
  return distplot

distplot(y, Yhat1)

# Polynomial regression

def polynomialRegression(x, y, degree):
  f = np.polyfit(x, y, degree)
  p = np.poly1d(f)
  return p

polynomialRegression(x, y, 3)

# Polynomial Regression with more than one Dimension
# We use 'preprocessing' library in scikit-learn

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)

x_polly = pr.fit_transform(df[['horsepower', 'curb-weight']])

# Pre-processing

from sklearn.preprocessing import StandardScaler

SCALE = StandardScaler()

SCALE.fit(df[['horsepower', 'highway-mpg']])

x_scale = SCALE.transform(df[['horsepower', 'highway-mpg']])

x_scale.shape

# Pipelines

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# we create a list of tuples
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2), 'model', LinearRegression())]

# Pipline constructor
pipe = Pipeline(Input)

# Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error

mean_squared_error(df['price'], Yhat1)

X1 = df[['highway-mpg']]
Y1 = df['price']

lm.fit(X1, Y1)

lm.score(X1, Y1)

# train_test_split()

from sklearn.model_selection import train_test_split

x_data = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y_data = df['price']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

def distplot(y_actual, y_predicted):
  ax1 = sns.distplot(y_actual, hist=False, color='r', label='Actual Value')
  sns.distplot(y_predicted, hist=False, color='b', label='Fitted Values', ax=ax1)
  return distplot

distplot(df['price'], x_train)

distplot(df['price'], x_test)

def Rsd(x_tr, x_te):
  Rsqu_test = []
  order = [1, 2, 3, 4]

  for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_tr)
    x_test_pr = pr.fit_transform(x_te)
    lm.fit(x_train_pr, y_train)
    Rsqu_test.append(lm.score(x_test_pr, y_test))
  return Rsqu_test

Rsd(x_train[['horsepower']], x_test[['horsepower']])

## Rigde Regression

from sklearn.linear_model import Ridge

RidgeModel = Ridge(alpha=0.1)

RidgeModel.fit(X, Y)

Yhat2 = RidgeModel.predict(X)

# Grid Search

from sklearn.model_selection import GridSearchCV

def Gridsearch(x_d, y_d, n):
  parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]
  RR = Ridge()
  Grid1 = GridSearchCV(RR, parameters1, cv=n)
  Grid1.fit(x_d, y_d)
  Grid1.best_estimator_
  scores = Grid1.cv_results_
  return scores['mean_test_score']

Gridsearch(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data, 4)

# Ridge with normalized function

def GridNormalize(x_d, y_d, n):
  parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100], 'normalize': [True, False]}]
  RR = Ridge()
  Grid1 = GridSearchCV(RR, parameters1, cv=n)
  Grid1.fit(x_d, y_d)
  Grid1.best_estimator_
  scores = Grid1.cv_results_
  return scores['mean_test_score']

GridNormalize(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data, 4)

param = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, param, cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
Grid1.best_estimator_
scores = Grid1.cv_results_
for param, mean_val, mean_test in zip(scores['params'], scores['mean_test_score'], scores['mean_score_time']):
  print(param, 'R^2 on test data: ', mean_val, 'R^2 on train data: ', mean_test)

