import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate x values
#x = np.linspace(-5, 5, 1000)
x = np.linspace(-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6)
# Compute activation function values
sigmoid_values = sigmoid(x)
relu_values = relu(x)
leaky_relu_values = leaky_relu(x)
tanh_values = tanh(x)

# Plot the activation functions
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_values, label='Sigmoid')
plt.title('Sigmoid Activation')
plt.grid()


plt.tight_layout()
plt.show()
