from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
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


#x = []
#y = []
# for k in range(1, 10):
#    km = KMeans(n_clusters=k)
#    km.fit(X)
#    x.append(k)
#    y.append(km.inertia_)

#plt.plot(x, y, marker='*')
# plt.show()
# 4

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
predictions = kmeans.predict(X_Test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
