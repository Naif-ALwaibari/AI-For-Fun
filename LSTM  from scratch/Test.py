import numpy as np
from LSTM_from_scratch import SimpleLSTMCell


print("--- Starting LSTM Test ---")

# Define Dimensions
INPUT_SIZE = 10
HIDDEN_SIZE = 4

# Initialize the model
lstm = SimpleLSTMCell(INPUT_SIZE, HIDDEN_SIZE)

# Create Dummy Data
x = np.random.randn(INPUT_SIZE, 1)       # Current input word
h_prev = np.zeros((HIDDEN_SIZE, 1))      # Previous hidden state
c_prev = np.zeros((HIDDEN_SIZE, 1))      # Previous cell state
target = np.ones((HIDDEN_SIZE, 1))       # Target output

# 1. Forward Pass
h_t, c_t, cache = lstm.forward(x, h_prev, c_prev)

# Calculate initial loss (Mean Squared Error)
initial_loss = np.mean((h_t - target)**2)
print(f"Initial Loss before training: {initial_loss:.4f}")

# 2. Compute the derivative of the loss (Simulated dh_next)
dh_next = h_t - target
dc_next = np.zeros((HIDDEN_SIZE, 1))

# 3. Backward Pass (Calculate Gradients)
dx, dh_prev, dc_prev, gradients = lstm.backward(cache, dh_next, dc_next)

# 4. Update Weights
lstm.update_weights(gradients, lr=0.5)

# 5. Verify Learning (Run forward pass again with updated weights)
h_t_new, _, _ = lstm.forward(x, h_prev, c_prev)
new_loss = np.mean((h_t_new - target)**2)

print(f"New Loss after 1 update step: {new_loss:.4f}")