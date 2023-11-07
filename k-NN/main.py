import random
import numpy as np
import pandas as pd
import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("WineDataset.csv")
print(df.describe())

df = df.dropna()
df = df.dropna(axis=1)
df = df.fillna(df.mean)

missing_data = df.isnull()

selected_columns = random.sample(range(len(df.columns) - 1), k=random.randint(0, df.shape[1] - 1))
selected_columns.append(len(df.columns) - 1)
random_df = df.iloc[:, selected_columns]

X = df[['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']].to_numpy()
y = df[['Wine']].to_numpy()
X_random = df.drop(columns=['Wine']).to_numpy()


indices = np.arange(len(X))
indices1 = np.arange(len(X_random))
np.random.shuffle(indices)
np.random.shuffle(indices1)
test_size = 0.2
split_index = int(len(X) * (1 - test_size))
split_index1 = int(len(X_random) * (1 - test_size))

X_train_ran, X_test_ran = X_random[indices1[:split_index1]], X_random[indices1[split_index1:]]
X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

model = KNeighborsClassifier.KNeighborsClassifier(3, 3)
model.fit(X_train, y_train, X_test)

pred = model.predict(3, y_test)
accuracy_score = model.accuracy(y_test, pred)
print("k = 3")
print(confusion_matrix(y_test, pred))
print(accuracy_score)
print("***************")

pred = model.predict(5, y_test)
accuracy_score = model.accuracy(y_test, pred)
print("k = 5")
print(confusion_matrix(y_test, pred))
print(accuracy_score)
print("***************")

pred = model.predict(10, y_test)
accuracy_score = model.accuracy(y_test, pred)
print("k = 10")
print(confusion_matrix(y_test, pred))
print(accuracy_score)
print("***************")

model_random = KNeighborsClassifier.KNeighborsClassifier(3, 3)
model_random.fit(X_train_ran, y_train, X_test_ran)

pred = model.predict(3, y_test)
accuracy_score = model.accuracy(y_test, pred)
print("k = 3, random")
print(confusion_matrix(y_test, pred))
print(accuracy_score)
print("***************")

pred = model.predict(5, y_test)
accuracy_score = model.accuracy(y_test, pred)
print("k = 3, random")
print(confusion_matrix(y_test, pred))
print(accuracy_score)
print("***************")

pred = model.predict(10, y_test)
accuracy_score = model.accuracy(y_test, pred)
print("k = 3, random")
print(confusion_matrix(y_test, pred))
print(accuracy_score)
print("***************")

