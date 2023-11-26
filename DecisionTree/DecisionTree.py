import numpy as np


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Индекс признака, по которому делается разделение
        self.threshold = threshold  # Порог для разделения
        self.value = value  # Значение, возвращаемое узлом (для листового узла)
        self.left = left  # Левый поддерево (меньшие или равные порогу значения)
        self.right = right  # Правый поддерево (большие значения)


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Максимальная глубина дерева
        self.tree = None  # Структура дерева

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Рекурсивное построение дерева
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Если достигнута максимальная глубина или все элементы принадлежат одному классу
        if depth == self.max_depth or len(unique_classes) == 1:
            return DecisionTreeNode(value=self._most_common_label(y))

        # Найдем наилучшее разделение
        best_feature_index, best_threshold = self._find_best_split(X, y)

        # Если не удалось найти разделение (все значения признака одинаковы)
        if best_feature_index is None:
            return DecisionTreeNode(value=self._most_common_label(y))

        # Разделение данных
        left_mask = X.iloc[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask

        # Рекурсивное построение левого и правого поддеревьев
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def _find_best_split(self, X, y):
        # Находим наилучшее разделение
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return None, None

        best_entropy = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(num_features):
            thresholds = np.unique(X.iloc[:, feature_index].values)
            for threshold in thresholds:
                left_mask = X.iloc[:, feature_index] <= threshold
                right_mask = ~left_mask
                entropy = self._calculate_entropy(y[left_mask], y[right_mask])
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _calculate_entropy(self, y_left, y_right):
        # Рассчет энтропии
        num_left = len(y_left)
        num_right = len(y_right)
        total_samples = num_left + num_right

        if total_samples == 0:
            return 0

        p_left = num_left / total_samples
        p_right = num_right / total_samples

        entropy_left = -sum((np.sum(y_left == c) / num_left) * np.log2(np.sum(y_left == c) / num_left)
                            for c in np.unique(y_left) if np.sum(y_left == c) > 0)
        entropy_right = -sum((np.sum(y_right == c) / num_right) * np.log2(np.sum(y_right == c) / num_right)
                             for c in np.unique(y_right) if np.sum(y_right == c) > 0)

        entropy = p_left * entropy_left + p_right * entropy_right
        return entropy

    def _most_common_label(self, y):
        if len(y) == 0:
            # Handle the case where the array is empty
            return None

        return np.bincount(y).argmax()

    def predict(self, X):
        # Прогнозирование меток для входных данных
        # return np.array([self._predict_tree(x, self.tree) for x in X])
        X_values = X.values
        # Y_values = Y.values
        prediction = []
        for row in X_values:
            prediction.append(self._predict_tree(row, self.tree))
        return prediction

    def _predict_tree(self, x, node):
        # Рекурсивное прогнозирование для одного элемента
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

    def get_tpr_fpr(self, div_value, Y_test, probas):
        predicted_values = []
        for i in range(len(probas)):
            if probas[i] >= div_value:
                predicted_values.append(1)
            else:
                predicted_values.append(0)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(Y_test)):
            if predicted_values[i] == 1:
                if Y_test.values[i] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if Y_test.values[i] == 1:
                    false_negative += 1
                else:
                    true_negative += 1

        true_positive_rate = (true_positive) / (true_positive + false_negative)
        false_positive_rate = (false_positive) / (false_positive + true_negative)
        return true_positive_rate, false_positive_rate

    def get_all_tprs(self, div_values, Y_test, probas):
        tprs = []
        for div_value in div_values:
            tprs.append(self.get_tpr_fpr(div_value, Y_test, probas)[0])
        return tprs

    def get_all_fprs(self, div_values, Y_test, probas):
        fprs = []
        for div_value in div_values:
            fprs.append(self.get_tpr_fpr(div_value, Y_test, probas)[1])
        return fprs

    def calculate_proba(self, X_values, root, depth=0):
        if root.value is not None:
            return 1 / depth
        if (X_values[root.feature_index] < root.threshold):
            return self.calculate_proba(X_values, root.left, depth + 1)
        else:
            return self.calculate_proba(X_values, root.right, depth + 1)

    def calculate_probas(self, X, model):
        X_values = X.values
        probas = []
        for row in X_values:
            probas.append(self.calculate_proba(row, self.tree))
        return probas

    def get_precision_recall(self, div_value, Y_test, probas):
        predicted_values = []
        for i in range(len(probas)):
            if probas[i] >= div_value:
                predicted_values.append(1)
            else:
                predicted_values.append(0)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(Y_test)):
            if predicted_values[i] == 1:
                if Y_test.values[i] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if Y_test.values[i] == 1:
                    false_negative += 1
                else:
                    true_negative += 1

        precision = (true_positive) / (true_positive + false_positive)
        recall = (true_positive) / (true_positive + false_negative)

        return precision, recall


    def get_all_precisions(self, div_values, Y_test, probas):
        precisions = []
        for div_value in div_values:
            precisions.append(self.get_precision_recall(div_value, Y_test, probas)[0])
        return precisions

    def get_all_recalls(self, div_values, Y_test, probas):
        recalls = []
        for div_value in div_values:
            recalls.append(self.get_precision_recall(div_value, Y_test, probas)[1])
        return recalls
