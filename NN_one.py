import numpy as np

# Activation function: tanh
def tanh(x):
    return np.tanh(x)

# Input values (you can adjust these if needed)
x1 = 0.9
x2 = 0.1
inputs = np.array([x1, x2])

# Random weights from interval [-0.5, 0.5]
np.random.seed(42)  # for reproducibility
w_h1 = np.random.uniform(-0.5, 0.5, size=2)
w_h2 = np.random.uniform(-0.5, 0.5, size=2)
w_out = np.random.uniform(-0.5, 0.5, size=2)

# Biases
b1 = 0.5
b2 = 0.7

# Calculate net input and output for hidden layer
net_h1 = np.dot(w_h1, inputs) + b1
out_h1 = tanh(net_h1)

net_h2 = np.dot(w_h2, inputs) + b2
out_h2 = tanh(net_h2)

# Hidden layer outputs
hidden_outputs = np.array([out_h1, out_h2])

# Output layer calculation
net_output = np.dot(w_out, hidden_outputs)
out_final = tanh(net_output)

print(out_final)


# Final output (sample run): -0.37096112247625995