import math
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Градиентный спуск
        for _ in range(self.iterations):
            # Предсказание
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Вычисление градиентов
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Обновление весов и смещения
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def accuracy(self, y_true, y_pred):
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
        total = len(y_true)
        return correct / total

    def log_loss(self, y_true, y_prob):
        epsilon = 1e-15  # используется для предотвращения логарифмической бесконечности
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)  # предотвращение выхода за границы [epsilon, 1-epsilon]

        # Вычисление Log Loss
        loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return loss

