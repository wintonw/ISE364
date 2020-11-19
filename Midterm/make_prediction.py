from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def penaltyScore(cm):
    return 50*cm[0][1] + 10*cm[1][0]


df = pd.read_csv(
    'https://github.com/wintonw/ISE364/raw/master/Midterm/project_data.csv')
df_obs = pd.read_csv(
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

rfc = RandomForestClassifier(n_estimators=1400, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', max_depth=90, bootstrap=False,)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_Test)
cm = confusion_matrix(y_test, predictions)
print(cm)
print(classification_report(y_test, predictions))

print(penaltyScore(cm))

# Format obs data for predictions
V0_obs = pd.get_dummies(df_obs['V0'], drop_first=True)
df_obs.drop(['V0'], axis=1, inplace=True)
df_obs = pd.concat([df_obs, V0_obs], axis=1)
df_obs.rename(columns={"B": "V0"}, inplace=True)

# Predict
obs_predictions = rfc.predict(df_obs)

# amend df
df_obs = pd.concat([df_obs, pd.Series(obs_predictions, name='target')], axis=1)

# save to csv
df_obs.to_csv('predictions.csv')
