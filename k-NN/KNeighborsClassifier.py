import pandas as pd
import numpy as np
from collections import Counter


class KNeighborsClassifier:
    def __init__(self, n_neighbors, p):
        self.n_neighbors = n_neighbors
        self.p = p

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, y, X_test):
        self.X_train = X
        self.y_train = y
        self.X_test = X_test

    def predict(self, k, y_test):
        y_test_predictions = []

        for test_row in self.X_test:
            distances = []

            for i in range(len(self.X_train)):
                distances.append((self.euclidean_distance(self.X_train[i], test_row), self.y_train[i]))

            distances.sort(key=lambda x: x[0])
            nearest_neighbors = distances[:k]

            # Собираем метки ближайших соседей
            neighbor_labels = [neighbor[1] for neighbor in nearest_neighbors]

            # Выбираем наиболее часто встречающийся класс
            most_common = Counter(tuple(neighbor_labels[i]) for i in range(len(neighbor_labels))).most_common(1)
            prediction = most_common[0][0]

            y_test_predictions.append(prediction)

        return np.array(y_test_predictions)

    def accuracy(self, y_true, y_pred):
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
        total = len(y_true)
        return correct / total
