import numpy as np

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

# Loading the data
data = np.loadtxt("prices.txt", delimiter=",")

# Divide the data into features and target variable
X = data[:, 0:-1]
y = data[:, -1]

# Feature Scaling
X, mean, std = normalize(X)

# Adding a column of ones to the features matrix for the bias term
X = np.hstack([np.ones([X.shape[0], 1]), X])

# Splitting the data into training and test sets
train_size = int(X.shape[0] * 0.75)
X_train, X_test = X[:train_size, :], X[train_size:, :]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize the model parameters with zeros
theta = np.zeros(X_train.shape[1])

# Define the cost function: Mean Squared Error
def cost_function(X, y, theta):
    return np.sum((X @ theta - y) ** 2) / (2 * len(y))

# Define the Gradient Descent function to minimize the cost function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= alpha * gradient
        cost = cost_function(X, y, theta)
        costs.append(cost)
    return theta, costs

# Run Gradient Descent
alpha = 0.01
iterations = 230
theta, costs = gradient_descent(X_train, y_train, theta, alpha, iterations)

# Print out the final cost
print("Final cost:", costs[-1])

# Define the prediction function
def predict(X, theta):
    return X @ theta

# Make predictions on the test set
y_test_pred = predict(X_test, theta)

# Calculate the mean squared error of the predictions
mse = np.mean((y_test - y_test_pred) ** 2)
print("Mean Squared Error on test set:", mse)
