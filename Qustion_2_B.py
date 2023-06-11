import numpy as np

# Load the data from the text file
data = np.genfromtxt('prices.txt', delimiter=',')
  
# Separate the features from the target (sale price)
X = data[:, :-1] # all columns except the last one
y = data[:, -1]  # only the last column

# Add a column of ones to X to represent the intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Split the data into a training set (75% of the data) and a testing set (25% of the data)
train_size = int(0.75 * X.shape[0])
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Standardize the features
X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], axis=0)) / np.std(X_train[:, 1:], axis=0)
X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_train[:, 1:], axis=0)) / np.std(X_train[:, 1:], axis=0)

# Initialize the model parameters (theta) to zero
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

# Run Gradient Descent with a larger learning rate and more iterations
alpha = 0.01
iterations = 5000
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

# Run Gradient Descent
alpha = 0.001
iterations = 2000
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


# Run Gradient Descent
alpha = 0.01
iterations = 10
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

# Run Gradient Descent
alpha = 0.001
iterations = 10
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
