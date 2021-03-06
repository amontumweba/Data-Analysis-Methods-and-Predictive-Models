# -*- coding: utf-8 -*-

!pip install yellowbrick

import pandas as pd
 import numpy as np

 import matplotlib.pyplot as plt
 from google.colab import drive

drive.mount('/gdrive')

house_data = pd.read_csv('/gdrive/MyDrive/Files/house_data_processed.csv')

house_data.shape

target = house_data['price']

features = house_data.drop('price', axis=1)

features.columns

from yellowbrick.target import FeatureCorrelation

feature_names = list(features.columns)

visualizer = FeatureCorrelation(labels=feature_names)

visualizer.fit(features, target)

visualizer.poof()

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression, mutual_info_regression

select_univariate = SelectKBest(f_regression, k=5).fit(features, target)

features_mask = select_univariate.get_support()

features_mask

selected_columns = features.columns[features_mask]

selected_columns

selected_features = features[selected_columns]

selected_features.head()

selected_features.describe()

from sklearn.preprocessing import scale

X = pd.DataFrame(data=scale(selected_features), columns=selected_features.columns)

y = target

X.describe()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

df = pd.DataFrame({'test': y_test, 'Predicted': y_pred})

df.head()

from sklearn.metrics import r2_score


score = linear_regression.score(X_train, y_train)
r2score = r2_score(y_test, y_pred)

print('Score: {}'.format(score))
print('r2_socre: {}'.format(r2score))

linear_regression.coef_

linear_regression.intercept_

import statsmodels.api as sm

import pandas.util.testing as tm

X_train = sm.add_constant(X_train)


model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_train)

print(model.summary())

select_univariate1 = SelectKBest(mutual_info_regression, k=5).fit(features, target)

features_mask1 = select_univariate1.get_support()

features_mask1

selected_columns1 = features.columns[features_mask1]

selected_columns1

selected_features1 = features[selected_columns1]

selected_features.head()

