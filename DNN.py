import numpy as np
from tensorflow.keras.datasets import mnist

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = (-1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Define the gradient of the cross-entropy loss function
def cross_entropy_gradient(y_true, y_pred):
    m = y_true.shape[0]
    grad = (1 / m) * (y_pred - y_true)
    return grad

# Define the deep neural network architecture
input_size = 784
hidden_size = 64
output_size = 10

# Initialize the weights and biases for each layer
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Define the learning rate and number of epochs
learning_rate = 0.1
num_epochs = 100

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255

# Convert the target data to one-hot encoding
y_train_onehot = np.zeros((y_train.shape[0], output_size))
y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
y_test_onehot = np.zeros((y_test.shape[0], output_size))
y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

# Train the deep neural network
for i in range(num_epochs):
    # Forward propagation
    hidden_layer = sigmoid(np.dot(X_train, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

    # Calculate the loss
    loss = cross_entropy_loss(y_train_onehot, output_layer)

    # Backpropagation
    output_delta = cross_entropy_gradient(y_train_onehot, output_layer) * sigmoid_derivative(output_layer)
    hidden_delta = np.dot(output_delta, W2.T) * sigmoid_derivative(hidden_layer)

    # Update the weights and biases
    W2 -= learning_rate * np.dot(hidden_layer.T, output_delta)
    b2 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(X_train.T, hidden_delta)
    b1 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    # Print the loss every 10 epochs
    if i % 10 == 0:
        print("Epoch", i, "Loss:", loss)

# Evaluate the deep neural network on the test set
hidden_layer = sigmoid(np.dot(X_test, W1) + b1)
output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
predictions = np.argmax(output_layer, axis=1)
accuracy = np.mean(predictions == y_test)