import numpy as np
import matplotlib.pyplot as plt


#feature-1
# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU Activation Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Tanh Activation Function
def tanh(x):
    return np.tanh(x)

# Generate x values
x = np.linspace(-10, 10, 1000)

# Plot the activation functions
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, relu(x), label='ReLU')
plt.title('ReLU Activation Function')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.title('Leaky ReLU Activation Function')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(x, tanh(x), label='Tanh')
plt.title('Tanh Activation Function')
plt.grid()

plt.tight_layout()
plt.show()
