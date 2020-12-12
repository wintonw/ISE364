from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def penaltyScore(cm):
    return 50*cm[0][1] + 10*cm[1][0]


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


param_grid = {'C': [0.1, 10, 100, 1000, 50], 'gamma': [
    1, 0.1, 0.01, 0.001, 0.0001, 0.005], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

print(grid.best_params_)
grid_predictions = grid.predict(X_Test)
cm = confusion_matrix(y_test, grid_predictions)
print(cm)
print(classification_report(y_test, grid_predictions))

print(penaltyScore(cm))