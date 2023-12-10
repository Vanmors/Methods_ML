import pandas as pd
import numpy as np
import LogisticRegression


def confusion_matrixHand(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Lengths of true and predicted labels must be the same."

    # Инициализация счетчиков
    true_negative, false_positive, false_negative, true_positive = 0, 0, 0, 0

    # Вычисление Confusion Matrix
    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == 0 and predicted_label == 0:
            true_negative += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 1 and predicted_label == 0:
            false_negative += 1
        elif true_label == 1 and predicted_label == 1:
            true_positive += 1

    # Возвращение результатов
    return true_negative, false_positive, false_negative, true_positive


df = pd.read_csv("diabetes.csv")

print(df)

mean_insulin = df['Insulin'].mean()
df['Insulin'] = df['Insulin'].replace(0, mean_insulin)

mean_insulin = df['BMI'].mean()
df['BMI'] = df['BMI'].replace(0, mean_insulin)

mean_insulin = df['SkinThickness'].mean()
df['SkinThickness'] = df['SkinThickness'].replace(0, mean_insulin)

mean_insulin = df['BloodPressure'].mean()
df['BloodPressure'] = df['BloodPressure'].replace(0, mean_insulin)

mean_insulin = df['Glucose'].mean()
df['Glucose'] = df['Glucose'].replace(0, mean_insulin)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

indices = np.arange(len(X))
np.random.shuffle(indices)
test_size = 0.2
split_index = int(len(X) * (1 - test_size))

X_train, X_test = X.iloc[indices[:split_index]], X.iloc[indices[split_index:]]
y_train, y_test = y.iloc[indices[:split_index]], y.iloc[indices[split_index:]]

model = LogisticRegression.LogisticRegression(0.001, 1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("accuracy:", model.accuracy(y_test.values, y_pred))
print("logloss:", model.log_loss(y_test.values, y_pred))

tn, fp, fn, tp = confusion_matrixHand(y_test, y_pred)

accuracy = (tp + tn) / (tp + tn + fp + fn)
try:
    precision = tp / (tp + fp)
except ZeroDivisionError:
    precision = 0
try:
    recall = tp / (tp + fn)
except ZeroDivisionError:
    recall = 0

f1_score = 2 * ((precision * recall) / (precision + recall));

print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)

print(df.columns)
