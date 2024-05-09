import numpy as np
class LinearRegressor:
    def __init__(self):
        self.coefficients = None
    
    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    def predict(self, X_new):
        X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        return X_new.dot(self.coefficients)

# Example usage:
# Let's assume X is your feature matrix and y is your target vector
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 3, 5])

# Creating an instance of LinearRegressor
model = LinearRegressor()
model.fit(X, y)
predictions = model.predict(X)
predictions
