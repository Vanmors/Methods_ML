import pandas as pd
import numpy as np
import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

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


for column_name, params in X_train.items():
  minimum = min(params)
  maximum = max(params)
  difference = maximum - minimum
  X_train[column_name] = (X_train[column_name] - minimum) / difference

for column_name, params in X_test.items():
      minimum = min(params)
      maximum = max(params)
      difference = maximum - minimum
      X_test[column_name] = (X_test[column_name] - minimum) / difference

print(X_train)


model = LogisticRegression.LogisticRegression(0.01, 1000)

model.newton_optimization(X_train, y_train)
y_pred = model.predict(X_test)

print("accuracy:", model.accuracy(y_test.values, y_pred))
# print("logloss:", model.log_loss(y_test.values, y_pred))

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

f1_score = 2 * ((precision * recall) / (precision + recall))

print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)

rates = [0.01, 0.2, 0.375, 0.5]
iterations = [100, 1000, 5000]

f1_scoreMax = 0
iterMax = 0
rateMax = 0
for rate in rates:
    for iteration in iterations:
        modelTest = LogisticRegression.LogisticRegression(rate, iteration)
        modelTest.fit(X_train, y_train)

        y_predTest = modelTest.predict(X_test)
        tn, fp, fn, tp = confusion_matrixHand(y_test, y_predTest)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        try:
            f1_score = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1_score = 0
        if f1_score > f1_scoreMax:
            f1_scoreMax = f1_score
            iterMax = iteration
            rateMax = rate

    for iteration in iterations:
        modelTest = LogisticRegression.LogisticRegression(rate, iteration)
        modelTest.newton_optimization(X_train, y_train)

        y_predTest = modelTest.predict(X_test)
        tn, fp, fn, tp = confusion_matrixHand(y_test, y_predTest)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        f1_score = 2 * ((precision * recall) / (precision + recall))
        if f1_score > f1_scoreMax:
            f1_scoreMax = f1_score
            iterMax = iteration
            rateMax = rate

print("best rate:", rateMax)
print("best iterate", iterMax)
print("best f1_score", f1_scoreMax)

print(df.describe())
summary_stats = df.describe()

# Выводим статистику
print("Статистика по датасету:")
print(summary_stats)

# Визуализируем статистику
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_stats.transpose(), palette="viridis")
plt.title("Статистика по датасету")
plt.ylabel("Значение")
plt.show()
