import numpy as np


class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_new):
        return np.array([self._predict(x_new) for x_new in X_new])

    def _predict(self, x_new):
        distance = np.sqrt(np.sum((self.X_train - x_new) ** 2, axis=1))
        # print(distance)
        k_index = np.argsort(distance)[:self.k]
        k_y = self.y_train[k_index]
        y_predict = np.mean(k_y)
        return y_predict


model = KNNRegressor()

X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
y_train = np.array([1, 2, 3, 4, 5])
model.fit(X_train, y_train)
X_new = np.array([[1.5, 1.5], [2.5, 2.5]])
y_predict = model.predict(X_new)
print(y_predict)
