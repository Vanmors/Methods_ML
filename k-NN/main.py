import numpy as np
import pandas as pd
import KNeighborsClassifier

df = pd.read_csv("WineDataset.csv")
print(df.describe())

df = df.dropna()
df = df.dropna(axis=1)
df = df.fillna(df.mean)

missing_data = df.isnull()

X = df[['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']].to_numpy()
y = df[['Wine']].to_numpy()

indices = np.arange(len(X))
np.random.shuffle(indices)
test_size = 0.2
split_index = int(len(X) * (1 - test_size))

X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

model = KNeighborsClassifier.KNeighborsClassifier(3, 3)
model.fit(X_train, y_train, X_test)
pred = model.predict(6, y_test)
print(pred)
accuracy_score = model.accuracy(y_test, pred)
print(y_test)
print(accuracy_score)
