# -*- coding: utf-8 -*-

import pandas as pd

from google.colab import drive

drive.mount('/gdrive')

data = pd.read_csv('/gdrive/MyDrive/Files/gender_voice_dataset.csv')

data.shape

data.describe().T

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
data['label'] = labelEncoder.fit_transform(data['label'].astype(str))

from sklearn.model_selection import train_test_split


features = data.drop('label', axis=1)
target = data['label']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

x_train.shape, y_train.shape

x_test.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logistic_model = LogisticRegression(penalty='l2', solver='liblinear')

logistic_model.fit(x_train, y_train)

y_pred = logistic_model.predict(x_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

print("Training score: ", logistic_model.score(x_train, y_train))

from sklearn.metrics import accuracy_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy: ', acc)
print('Precision Score: ', pre)
print('Recall Score: ', recall)

# Feature selection

from yellowbrick.target import FeatureCorrelation

feature_names = list(features.columns)

visualizer = FeatureCorrelation(labels = feature_names)

visualizer.fit(features, target)

visualizer.poof()

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest

select_univariate = SelectKBest(mutual_info_classif, k=4).fit(features, target)

features_mask = select_univariate.get_support()

features_mask

selected_columns = features.columns[features_mask]

selected_columns

selected_features = features[selected_columns]

selected_features.head()

x_train1, x_test1, y_train1, y_test1 = train_test_split(selected_features, target, test_size=.2)

logistic_model1 = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model1.fit(x_train1, y_train1)

y_pred1 = logistic_model1.predict(x_test1)

acc1 = accuracy_score(y_test1, y_pred1)
pre1 = precision_score(y_test1, y_pred1)
recall1 = recall_score(y_test1, y_pred1)

print('Accuracy: ', acc1)
print('Precision Score: ', pre1)
print('Recall Score: ', recall1)

