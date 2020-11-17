from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    'https://github.com/wintonw/ISE364/raw/master/Midterm/project_data.csv')
dfObs = pd.read_csv(
    'https://github.com/wintonw/ISE364/raw/master/Midterm/new_obs.csv')


# V0 to binary
V0 = pd.get_dummies(df['V0'], drop_first=True)
df.drop(['V0'], axis=1, inplace=True)
df = pd.concat([df, V0], axis=1)


# split the data
X = df.drop('target', axis=1)
y = df.target
X_train, X_Test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=12)

linearSVC = LinearSVC()
linearSVC.fit(X_Test, y_test)

linearSVC_predictions = linearSVC.predict(X_Test)
print(confusion_matrix(y_test, linearSVC_predictions))
print(classification_report(y_test, linearSVC_predictions))
