import numpy as np

class LinearRegressor:
    def __init__(self):
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Slope

    def fit(self, X, y):
        # Ensure X and y are numpy arrays and have the correct shape
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Calculate the coefficients using the normal equation
        # beta = (X'X)^-1 * X'y
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
        beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.beta_0 = beta[0]
        self.beta_1 = beta[1:]

    def predict(self, X_new):
        # Ensure X_new is a numpy array and has the correct shape
        X_new = np.array(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        # Predict using the model's coefficients
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]  # add x0 = 1 to each instance
        return X_new_b.dot(np.r_[self.beta_0, self.beta_1])


# Example usage
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
lin_reg = LinearRegressor()
lin_reg.fit(X, y)
X_new = np.array([[0], [2]])
y_pred = lin_reg.predict(X_new)
print(y_pred)
