import numpy as np

# Load the data from the txt file
data = np.genfromtxt('prices.txt', delimiter=',')

# Normalize the data
# data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Split the data
np.random.shuffle(data)
train_data = data[:int(0.75*len(data))]  # 75% of data for training
test_data = data[int(0.25*len(data)):]   # 25% of data for testing

X_train = train_data[:,:-1]# Load the data from the txt file -1 means the last column
y_train = train_data[:,-1]# Load the data from the txt file -1 means the last column

X_test = test_data[:,:-1]
y_test = test_data[:,-1]

# Implementing the Linear Regression

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

B = np.zeros(X_train.shape[1])
alpha = 0.005
iter_ = 2000

newB, cost_history = gradient_descent(X_train, y_train, B, alpha, iter_)

# Testing the model

y_pred = X_test.dot(newB)

# Now y_pred contains predicted prices for the apartments in the test set.


# Calculate the mean squared error of the predictions
mse = np.sum((y_pred - y_test) ** 2) / len(y_test)
print('Mean Squared Error:', mse)

