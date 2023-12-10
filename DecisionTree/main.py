import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import DecisionTree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import NotBinaryTree



# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score


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

# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

data = pd.concat([X, y], axis=1)
# metadata
# print(mushroom.metadata)
#
# variable information
# print(mushroom.variables)


unique_values = []
for i in X['stalk-root'].values:
    if i not in unique_values and i is not None:
        unique_values.append(i)

X['stalk-root'].fillna(np.random.choice(unique_values), inplace=True)


all_features = X.columns

num_features_to_select = len(X.columns) - int(np.sqrt(len(all_features)))


selected_features_to_drop = np.random.choice(all_features, size=num_features_to_select, replace=False)
X = X.drop(selected_features_to_drop, axis=1)

# кодируем категориальные признаки
columns = X.columns.values
X = data[columns].apply(lambda col: col.astype('category').cat.codes)
y = data['poisonous'].astype('category').cat.codes

indices = np.arange(len(X))
np.random.shuffle(indices)
test_size = 0.2
split_index = int(len(X) * (1 - test_size))

# X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
# y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = NotBinaryTree.DecisionTree(4)
model.fit(X_train, y_train)

X_test = X_test.astype('int8')

y_pred = model.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(recall_score(y_test, y_pred))

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

tn, fp, fn, tp = confusion_matrixHand(y_test, y_pred)
print("True Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive:", tp)

print("accuracy:", (tp + tn) / (tp + tn + fp + fn))
print("precision:", tp / (tp + fp))
print("recall:", tp / (tp + fn))


probas = np.sort(model.calculate_probas(X_test, model))
unique_probas = np.unique(probas)


tprs = model.get_all_tprs(np.unique(probas), y_test, probas)
fprs = model.get_all_fprs(np.unique(probas), y_test, probas)



auc_roc = 0
for i in range(len(fprs) - 1):
    s = (tprs[i] + tprs[i+1]) / 2 * (fprs[i + 1] - fprs[i])
    auc_roc += s
print(f'auc_roc = {abs(auc_roc)}')


precisions = model.get_all_precisions(np.unique(probas), y_test, probas)
recalls = model.get_all_recalls(np.unique(probas), y_test, probas)

auc_pr = 0
for i in range(len(fprs) - 1):
    s = (precisions[i] + precisions[i+1]) / 2 * (recalls[i + 1] - recalls[i])
    auc_pr += s
print(f'auc_roc = {abs(auc_pr)}')

plt.title('ROC curve', fontsize=12)
plt.xlabel('FPR', fontsize=8)
plt.ylabel('TPR', fontsize=8)
plt.plot(fprs, tprs, 'b', fprs, fprs, 'r')
plt.xlim(0, 1.2)
plt.ylim(0, 1.2)
plt.show()

plt.title('PR curve', fontsize=12)
plt.xlabel('recall', fontsize=8)
plt.ylabel('precision', fontsize=8)
plt.plot(recalls, precisions, 'b', recalls, [1 - i for i in recalls], 'r')
plt.xlim(0, 1.2)
plt.ylim(0, 1.2)
plt.show()
