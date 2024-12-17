import numpy as np
import matplotlib.pyplot as plt

# Generate a simple input signal (sinusoidal)
t = np.linspace(0, 10, 1000)
input_signal = np.sin(2 * np.pi * 0.5 * t)

# Reservoir parameters
n_neurons = 50
reservoir_weights = np.random.randn(n_neurons, n_neurons) * 0.1  # sparse random connections
input_weights = np.random.randn(n_neurons) * 0.5

# Intrinsic plasticity parameters
gain = np.ones(n_neurons)  # initial gain
bias = np.zeros(n_neurons)  # initial bias
target_mean = 0.5  # target mean firing rate
target_std = 0.2   # target standard deviation

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Intrinsic plasticity update rule
def update_intrinsic_plasticity(output, gain, bias, lr=0.8):
    error_mean = target_mean - np.mean(output)
    error_std = target_std - np.std(output)
    
    gain += lr * error_std * output * (1 - output)  # update gain based on std error
    bias += lr * error_mean * output * (1 - output)  # update bias based on mean error
    
    return gain, bias

# Simulate the reservoir without intrinsic plasticity
reservoir_outputs_no_ip = []
state_no_ip = np.zeros(n_neurons)

for i in range(len(input_signal)):
    input_to_reservoir = input_signal[i] * input_weights
    pre_activation = np.dot(reservoir_weights, state_no_ip) + input_to_reservoir
    state_no_ip = sigmoid(pre_activation)  # No gain/bias adaptation
    reservoir_outputs_no_ip.append(state_no_ip)

# Simulate the reservoir with intrinsic plasticity
reservoir_outputs_ip = []
state_ip = np.zeros(n_neurons)
gain = np.ones(n_neurons)  # Reset gain
bias = np.zeros(n_neurons)  # Reset bias

for i in range(len(input_signal)):
    input_to_reservoir = input_signal[i] * input_weights
    pre_activation = np.dot(reservoir_weights, state_ip) + input_to_reservoir
    state_ip = sigmoid(gain * pre_activation + bias)
    reservoir_outputs_ip.append(state_ip)
    gain, bias = update_intrinsic_plasticity(state_ip, gain, bias)

# Plot results
plt.figure(figsize=(12, 10))

# Input signal
plt.subplot(3, 1, 1)
plt.plot(t, input_signal, label="Input Signal", color='blue')
plt.title("Input Signal (Sine Wave)")
plt.grid(True)

# Without Intrinsic Plasticity
plt.subplot(3, 1, 2)
plt.plot(t, np.array(reservoir_outputs_no_ip))
plt.title("Reservoir Outputs Without Intrinsic Plasticity (First 5 Neurons)")
plt.grid(True)

# With Intrinsic Plasticity
plt.subplot(3, 1, 3)
plt.plot(t, np.array(reservoir_outputs_ip))
plt.title("Reservoir Outputs With Intrinsic Plasticity (First 5 Neurons)")
plt.grid(True)

plt.tight_layout()
plt.show()
