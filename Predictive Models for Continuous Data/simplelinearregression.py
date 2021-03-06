# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive

drive.mount('/gdrive')

house_data = pd.read_csv('/gdrive/MyDrive/Files/kc_house_data.csv')

house_data.drop(['id', 'lat', 'long', 'zipcode'], inplace=True, axis=1)

# The date info gives us date on which that particular house was sold at that price recorded
house_data['date'] = pd.to_datetime(house_data['date'])
house_data['house_age'] = house_data['date'].dt.year - house_data['yr_built']


house_data.drop('date', inplace=True, axis=1)
house_data = house_data.drop('yr_built', axis=1)

house_data['renovated'] = house_data['yr_renovated'].apply(lambda x:0 if x == 0 else 1)

house_data.drop('yr_renovated', inplace=True, axis=1)

house_data[['renovated', 'house_age']].sample(10)

house_data.to_csv('/gdrive/MyDrive/Files/house_data_processed.csv', index=False)

sns.lmplot('sqft_living', 'price', house_data)
plt.show()

sns.lmplot('house_age', 'price', house_data)
plt.show()

sns.lmplot('floors', 'price', house_data)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = house_data[['sqft_living']]
y = house_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

linear_regression = LinearRegression()

model = linear_regression.fit(X_train, y_train)
y_pred = model.predict(X_test)

df = pd.DataFrame({'test': y_test, 'predicted': y_pred})

df.sample(10)

plt.figure(figsize=(10, 8))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, c='r')

plt.show()

print('Training score: ', linear_regression.score(X_train, y_train))

from sklearn.metrics import r2_score

score =r2_score(y_test, y_pred)

print('Testing score: ', score)

theta_0 = linear_regression.coef_

theta_0

intercept = linear_regression.intercept_

intercept

plt.subplots(figsize=(10, 8))

plt.plot(y_pred, label='Prediction')
plt.plot(y_test.values, label='Actual')

plt.legend()

plt.show()

import statsmodels.api as sm

import pandas.util.testing as tm

X_train[:5]

X_train = sm.add_constant(X_train)

X_train[:5]

model = sm.OLS(y_train, X_train).fit()

print(model.summary())

theta_0, intercept

