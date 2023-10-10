import numpy as np
import pandas as pd
import LinearRegression

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_total)
    return r2


df = pd.read_csv("california_housing_train.csv")
print(df.describe())

df = df.dropna()
df = df.dropna(axis=1)
df = df.fillna(df.mean)

missing_data = df.isnull()
X = df[['longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income']].to_numpy()
# X = df[['longitude', 'latitude']].to_numpy()
X1 = df[['longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms']].to_numpy()
X2 = df[['longitude', 'latitude']].to_numpy()
y = df[['median_house_value']].to_numpy()

indices = np.arange(len(X))
indices1 = np.arange(len(X1))
indices2 = np.arange(len(X2))
np.random.shuffle(indices)
np.random.shuffle(indices1)
np.random.shuffle(indices2)

# Указание доли данных, которые должны быть выделены в тестовую выборку
test_size = 0.2
split_index = int(len(X) * (1 - test_size))
split_index1 = int(len(X1) * (1 - test_size))
split_index2 = int(len(X2) * (1 - test_size))

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

X1_train, X1_test = X1[indices1[:split_index1]], X1[indices1[split_index1:]]
y1_train, y1_test = y[indices1[:split_index1]], y[indices1[split_index1:]]

X2_train, X2_test = X2[indices2[:split_index2]], X2[indices2[split_index2:]]
y2_train, y2_test = y[indices2[:split_index2]], y[indices2[split_index2:]]

model = LinearRegression.Linear_Regression()

model.fit(X, y)
predicted_y = model.predict(X_test)

model.fit(X1, y)
predicted_y1 = model.predict(X1_test)

model.fit(X2, y)
predicted_y2 = model.predict(X2_test)

print("Для первой модели(все признаки) ", r_squared(y_test, predicted_y))
print("Для первой модели(все, кроме последнийх трёх) ", r_squared(y1_test, predicted_y1))
print("Для первой модели(только местоположение) ", r_squared(y2_test, predicted_y2))
