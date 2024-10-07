import numpy as np

class StackedEchoStateNetworkWithIP:
    def __init__(self, input_size, reservoir_sizes, output_size, spectral_radius=0.9, target_mean=0, target_std=1, lr=0.001):
        self.num_layers = len(reservoir_sizes)
        self.spectral_radius = spectral_radius
        self.target_mean = target_mean  # Target mean for activations
        self.target_std = target_std    # Target standard deviation for activations
        self.lr = lr  # Learning rate for intrinsic plasticity
        
        # Initialize the reservoirs and their intrinsic plasticity parameters
        self.W_in = []
        self.W_res = []
        self.gains = []
        self.biases = []
        
        for i, reservoir_size in enumerate(reservoir_sizes):
            in_size = input_size if i == 0 else reservoir_sizes[i - 1]
            W_in = np.random.uniform(-1, 1, (reservoir_size, in_size))
            W_res = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
            W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W_res)))
            
            self.W_in.append(W_in)
            self.W_res.append(W_res)
            
            # Initialize gain and bias for each layer
            self.gains.append(np.ones(reservoir_size))
            self.biases.append(np.zeros(reservoir_size))
        
        # Output weight (connecting final reservoir to output)
        self.W_out = np.random.uniform(-1, 1, (reservoir_sizes[-1], output_size))  # Change shape to (100, 1)

    def _intrinsic_plasticity_update(self, neuron_output, layer):
        """
        Update the gain and bias of each neuron in the current layer using intrinsic plasticity.
        """
        delta_bias = self.lr * (self.target_mean - neuron_output)  # Update bias
        delta_gain = self.lr * (1 / self.gains[layer] + (neuron_output - self.target_mean) * (neuron_output - self.target_mean - self.target_std**2))
        self.biases[layer] += delta_bias  # Update bias term
        self.gains[layer] += delta_gain  # Update gain term

    def _update_reservoir(self, X, reservoir_state, layer):
        """
        Update reservoir for a given layer, applying gain and bias per neuron for intrinsic plasticity.
        """
        # Compute pre-activation (before applying gain and bias)
        pre_activation = np.dot(self.W_in[layer], X) + np.dot(self.W_res[layer], reservoir_state)
        
        # Apply gain and bias to each neuron's pre-activation
        reservoir_activation = np.tanh(self.gains[layer] * pre_activation + self.biases[layer])

        # Update gain and bias using intrinsic plasticity
        self._intrinsic_plasticity_update(reservoir_activation, layer)
        
        return reservoir_activation

    def fit(self, X_train, Y_train):
        # Initialize reservoir states for each layer
        reservoir_states = [np.zeros((X_train.shape[0], size)) for size in reservoir_sizes]
        reservoir_state = [np.zeros(size) for size in reservoir_sizes]
        
        # Loop through each time step
        for t in range(X_train.shape[0]):
            input_signal = X_train[t]
            
            # Update each reservoir layer
            for layer in range(self.num_layers):
                reservoir_state[layer] = self._update_reservoir(input_signal, reservoir_state[layer], layer)
                reservoir_states[layer][t] = reservoir_state[layer]
                
                # The input to the next layer is the output of the current one
                input_signal = reservoir_state[layer]
        
        # Train the readout layer using the final reservoir layer's states
        self.W_out = np.dot(np.linalg.pinv(reservoir_states[-1]), Y_train)

    def predict(self, X_test):
        Y_pred = np.zeros(X_test.shape)
        reservoir_state = [np.zeros(size) for size in reservoir_sizes]

        for t in range(X_test.shape[0]):
            input_signal = X_test[t]
            
            # Update each layer in sequence
            for layer in range(self.num_layers):
                reservoir_state[layer] = self._update_reservoir(input_signal, reservoir_state[layer], layer)
                input_signal = reservoir_state[layer]  # Pass to the next layer
            
            # Compute output for the final layer
            Y_pred[t] = np.dot(reservoir_state[-1], self.W_out).flatten()  # No need to reshape reservoir_state[-1]
        
        return Y_pred

# Example usage with a sine wave time series
X = np.sin(np.linspace(0, 8 * np.pi, 200)).reshape(-1, 1)

# Training data: input sequence X_train and target sequence Y_train
X_train = X[:150]
Y_train = X[1:151]

# Define a stacked ESN with two layers of 100 neurons each
reservoir_sizes = [20, 30, ]
esn_stacked_ip = StackedEchoStateNetworkWithIP(input_size=1, reservoir_sizes=reservoir_sizes, output_size=1)
esn_stacked_ip.fit(X_train, Y_train)

# Test on unseen data
X_test = X[150:199]
Y_pred = esn_stacked_ip.predict(X_test)

# Output predicted values for analysis
print(Y_pred)

# Plot the original and predicted sine wave
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(X, label='Original')
plt.plot(np.arange(150, 199), Y_pred, label='Predicted')
plt.legend()
plt.show()