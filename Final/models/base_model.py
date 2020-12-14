import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
start_time = time.time()

# read the data
df = pd.read_csv('/home/ubuntu/ISE364/Final/data.csv', usecols=[
                 *range(0, 12)], names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# each unique values get a col
ohe = OneHotEncoder()
feature_arr = ohe.fit_transform(df[[1]]).toarray()
feature_label = [str(1) + "_" + str(i) for i in range(feature_arr.shape[1])]
feature = pd.DataFrame(feature_arr, columns=feature_label)

# cols
columns = [*range(0, 12)]
categorical_columns = [1, 3, 5, 6, 7]
# if data is categorical, transform them
df_ohe = df.copy()
for index, i in enumerate(columns):
    if index in categorical_columns:
        ohe = OneHotEncoder()
        feature_arr = ohe.fit_transform(df[[index]]).toarray()
        feature_label = [str(index) + "_" + str(k)
                         for k in range(feature_arr.shape[1])]
        feature = pd.DataFrame(feature_arr, columns=feature_label)
        # drop the last col
        # feature = feature.iloc[:, :-1]
        df_ohe = pd.concat([df_ohe, feature], axis=1)
        df_ohe.drop(index, axis=1, inplace=True)

# set up X, y
X = df_ohe.drop(11, axis=1).values
y = df_ohe[11].values
# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=12)

# scale the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model set up
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', patience=50, verbose=1)


model = Sequential()
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
# loss for binary
model.compile(loss='binary_crossentropy', optimizer='adam')

# fit
model.fit(x=X_train, y=y_train, epochs=1000, validation_data=(
    X_test, y_test), verbose=0, callbacks=early_stop)

# plot
model_loss = pd.DataFrame(model.history.history)
model_loss.plot().figure.savefig('base_loss.png')


# metrics
predictions = model.predict_classes(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("accuracy_score", accuracy_score(y_test, predictions))

print("\n This took", time.time() - start_time, "seconds to run")
