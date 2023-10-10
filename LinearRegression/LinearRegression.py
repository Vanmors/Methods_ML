import numpy as np


class Linear_Regression():
    def fit(self, X, y):
        self.X = X
        self.y = y
        n, m = X.shape  # n - количество наблюдений, m - количество признаков

        self.beta = np.zeros(m)
        X_with_intercept = np.column_stack([np.ones(n), X])
        self.beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    def predict(self, X):
        n, m = X.shape
        X_with_intercept = np.column_stack([np.ones(n), X])
        Y_predicted = np.dot(X_with_intercept, self.beta)
        return Y_predicted
