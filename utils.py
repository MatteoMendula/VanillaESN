import numpy as np

def calculate_f1_score(series_a, series_b):
    # Convert to lists if not already
    series_a = list(series_a)
    series_b = list(series_b)

    # Initialize counts
    TP = TN = FP = FN = 0

    # Calculate TP, TN, FP, FN
    for a, b in zip(series_a, series_b):
        if a == 1 and b == 1:
            TP += 1
        elif a == 0 and b == 0:
            TN += 1
        elif a == 0 and b == 1:
            FP += 1
        elif a == 1 and b == 0:
            FN += 1

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score


class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, random_seed=42):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size))
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask
        spectral_radius_current = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / spectral_radius_current) * spectral_radius
        self.W_out = None
    
    def _update_reservoir(self, X, prev_reservoir_state):
        X = X.reshape(-1, 1)
        prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
        pre_activation = np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)
        return np.tanh(pre_activation)
    
    def fit(self, X_train, y_train, warm_up_steps=10, reservoir_state=None):
        n_samples = X_train.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))

        # Warm-up phase
        for t in range(warm_up_steps, n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)

        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            reservoir_states[t] = reservoir_state.ravel()
        extended_states = np.hstack([reservoir_states, X_train])
        self.W_out = np.dot(np.linalg.pinv(extended_states), y_train)
    
    def predict(self, X_test, reservoir_state=None):
        n_samples = X_test.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        predictions = np.zeros(n_samples)
        for t in range(n_samples):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            pred = np.dot(extended_state, self.W_out)
            predictions[t] = 1 if pred > 0.5 else 0  # Binary classification decision boundary at 0.5
        return predictions

class EchoStateNetworkWithIP:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, random_seed=42, target_mean=0.2, target_var=0.1, ip_learning_rate=0.01):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size)) * 0.1  # Scale down input weights
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask
        spectral_radius_current = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / spectral_radius_current) * spectral_radius
        self.W_out = None

        # Intrinsic plasticity parameters
        self.a = np.ones((reservoir_size, 1))  # Slopes
        self.b = np.zeros((reservoir_size, 1))  # Biases
        self.target_mean = target_mean
        self.target_var = target_var
        self.ip_learning_rate = ip_learning_rate  # Adjustable learning rate for IP
    
    # def _update_reservoir(self, X, prev_reservoir_state):
    #     X = X.reshape(-1, 1)
    #     prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
    #     pre_activation = np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)

    #     # Apply intrinsic plasticity adaptation
    #     reservoir_state = np.tanh(pre_activation)
    #     reservoir_state = self.a * reservoir_state + self.b
    #     self._update_intrinsic_plasticity(reservoir_state)

    #     # Clip reservoir state to prevent overflow
    #     reservoir_state = np.clip(reservoir_state, -5, 5)
    #     return np.tanh(reservoir_state)  # Final activation

    def _update_reservoir(self, X, prev_reservoir_state):
        X = X.reshape(-1, 1)
        prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
        pre_activation = self.a * (np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)) + self.b

        # Apply intrinsic plasticity adaptation
        reservoir_state = np.tanh(pre_activation)
        reservoir_state = self.a * reservoir_state + self.b
        self._update_intrinsic_plasticity(reservoir_state)

        # Clip reservoir state to prevent overflow
        reservoir_state = np.clip(reservoir_state, -5, 5)
        return np.tanh(reservoir_state)  # Final activation
    
    # def _update_intrinsic_plasticity(self, reservoir_state):
    #     # Ensure numerical stability in IP updates
    #     mean = np.mean(reservoir_state)
    #     variance = np.var(reservoir_state)

    #     # Avoid divide-by-zero and overflow in updates
    #     if np.isfinite(mean) and np.isfinite(variance):
    #         self.a += self.ip_learning_rate * ((1 / max(self.target_var, 1e-8)) * (variance - self.target_var))
    #         self.b += self.ip_learning_rate * (self.target_mean - mean)

    #         # Clip a and b to reasonable ranges to prevent overflow
    #         self.a = np.clip(self.a, 0.01, 10)
    #         self.b = np.clip(self.b, -5, 5)

    def _update_intrinsic_plasticity(self, reservoir_state):
        mu = self.target_mean
        sigma = np.sqrt(self.target_var)
        eta = self.ip_learning_rate

        # Flatten reservoir state for computation
        x = reservoir_state.ravel()  # Shape (200,) for 1D neurons

        # Compute bias update delta_b
        delta_b = -eta * ((-mu / (sigma ** 2)) + (x / (sigma ** 2)) + (1 - x ** 2) + mu * x)
        delta_b = delta_b.reshape(self.b.shape)  # Match shape (200, 1)

        # Compute gain update delta_g
        x_net = np.dot(self.W, x.reshape(-1, 1))  # Net input to reservoir
        delta_g = eta / self.a + delta_b * x_net
        delta_g = delta_g.reshape(self.a.shape)  # Match shape (200, 1)

        # Update gain and bias
        self.b += delta_b
        self.a += delta_g

        # Ensure numerical stability
        self.a = np.clip(self.a, 0.01, 10)
        self.b = np.clip(self.b, -5, 5)

    def fit(self, X_train, y_train, warm_up_steps=10, reservoir_state=None):
        n_samples = X_train.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))

        # Warm-up phase
        for t in range(warm_up_steps, n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)

        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            reservoir_states[t] = reservoir_state.ravel()
        extended_states = np.hstack([reservoir_states, X_train])

        # Check for NaN or Inf in extended states
        if not np.all(np.isfinite(extended_states)):
            raise ValueError("Non-finite values detected in extended states. Check IP or reservoir initialization.")

        self.W_out = np.dot(np.linalg.pinv(extended_states), y_train)
    
    def predict(self, X_test, reservoir_state=None):
        n_samples = X_test.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        predictions = np.zeros(n_samples)
        for t in range(n_samples):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            pred = np.dot(extended_state, self.W_out)
            # save pred on txt file
            with open("pred.txt", "a") as f:
                f.write(str(pred) + "\n")
            predictions[t] = 1 if pred > 0.5 else 0  # Binary classification decision boundary at 0.5
        return predictions

class _EchoStateNetworkWithIP_FedIp:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, random_seed=42, target_mean=0.2, target_var=0.1, ip_learning_rate=0.01):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size)) * 0.1  # Scale down input weights
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask
        spectral_radius_current = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / spectral_radius_current) * spectral_radius
        self.W_out = None

        # Intrinsic plasticity parameters
        self.a = np.ones((reservoir_size, 1))  # Slopes (gain)
        self.b = np.zeros((reservoir_size, 1))  # Biases
        self.target_mean = target_mean
        self.target_var = target_var
        self.ip_learning_rate = ip_learning_rate  # Adjustable learning rate for IP

    def set_ip_parameters(self, a, b):
        """
        Set the gain (a) and bias (b) for intrinsic plasticity.
        Args:
            a (numpy.ndarray): Gain parameters (should have shape (reservoir_size, 1)).
            b (numpy.ndarray): Bias parameters (should have shape (reservoir_size, 1)).
        """
        if a.shape != self.a.shape or b.shape != self.b.shape:
            raise ValueError("Shape mismatch: a and b must match the reservoir size.", a.shape, self.a.shape, b.shape, self.b.shape)
        self.a = np.copy(a)
        self.b = np.copy(b)

    def get_ip_parameters(self):
        """
        Retrieve the gain (a) and bias (b) for intrinsic plasticity.
        Returns:
            tuple: A tuple (a, b) where both are numpy arrays of shape (reservoir_size, 1).
        """
        return np.copy(self.a), np.copy(self.b)

    def _update_reservoir(self, X, prev_reservoir_state):
        X = X.reshape(-1, 1)
        prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
        pre_activation = np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)

        # Apply intrinsic plasticity adaptation
        reservoir_state = np.tanh(pre_activation)
        reservoir_state = self.a * reservoir_state + self.b
        self._update_intrinsic_plasticity(reservoir_state)

        # Clip reservoir state to prevent overflow
        reservoir_state = np.clip(reservoir_state, -5, 5)
        return np.tanh(reservoir_state)  # Final activation
    
    def _update_intrinsic_plasticity(self, reservoir_state):
        # Ensure numerical stability in IP updates
        mean = np.mean(reservoir_state)
        variance = np.var(reservoir_state)

        # Avoid divide-by-zero and overflow in updates
        if np.isfinite(mean) and np.isfinite(variance):
            self.a += self.ip_learning_rate * ((1 / max(self.target_var, 1e-8)) * (variance - self.target_var))
            self.b += self.ip_learning_rate * (self.target_mean - mean)

            # Clip a and b to reasonable ranges to prevent overflow
            self.a = np.clip(self.a, 0.01, 10)
            self.b = np.clip(self.b, -5, 5)

    def fit(self, X_train, y_train, warm_up_steps=10, reservoir_state=None):
        n_samples = X_train.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))

        # Warm-up phase
        for t in range(warm_up_steps, n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)

        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            reservoir_states[t] = reservoir_state.ravel()
        extended_states = np.hstack([reservoir_states, X_train])

        # Check for NaN or Inf in extended states
        if not np.all(np.isfinite(extended_states)):
            raise ValueError("Non-finite values detected in extended states. Check IP or reservoir initialization.")

        self.W_out = np.dot(np.linalg.pinv(extended_states), y_train)
    
    def predict(self, X_test, reservoir_state=None):
        n_samples = X_test.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        predictions = np.zeros(n_samples)
        for t in range(n_samples):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            pred = np.dot(extended_state, self.W_out)
            predictions[t] = 1 if pred > 0.5 else 0  # Binary classification decision boundary at 0.5
        return predictions


class EchoStateNetworkWithIP_FedIp:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, random_seed=42, target_mean=0.2, target_var=0.1, ip_learning_rate=0.01):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size)) * 0.1  # Scale down input weights
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask
        spectral_radius_current = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / spectral_radius_current) * spectral_radius
        self.W_out = None

        # Intrinsic plasticity parameters
        self.a = np.ones((reservoir_size, 1))  # Slopes (gain)
        self.b = np.zeros((reservoir_size, 1))  # Biases
        self.target_mean = target_mean
        self.target_var = target_var
        self.ip_learning_rate = ip_learning_rate  # Adjustable learning rate for IP

    def set_ip_parameters(self, a, b):
        """
        Set the gain (a) and bias (b) for intrinsic plasticity.
        Args:
            a (numpy.ndarray): Gain parameters (should have shape (reservoir_size, 1)).
            b (numpy.ndarray): Bias parameters (should have shape (reservoir_size, 1)).
        """
        if a.shape != self.a.shape or b.shape != self.b.shape:
            raise ValueError("Shape mismatch: a and b must match the reservoir size.", a.shape, self.a.shape, b.shape, self.b.shape)
        self.a = np.copy(a)
        self.b = np.copy(b)

    def get_ip_parameters(self):
        """
        Retrieve the gain (a) and bias (b) for intrinsic plasticity.
        Returns:
            tuple: A tuple (a, b) where both are numpy arrays of shape (reservoir_size, 1).
        """
        return np.copy(self.a), np.copy(self.b)

    def _update_reservoir(self, X, prev_reservoir_state):
        X = X.reshape(-1, 1)
        prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
        pre_activation = np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)

        # Apply intrinsic plasticity adaptation
        reservoir_state = np.tanh(pre_activation)
        reservoir_state = self.a * reservoir_state + self.b
        self._update_intrinsic_plasticity(reservoir_state)

        # Clip reservoir state to prevent overflow
        reservoir_state = np.clip(reservoir_state, -5, 5)
        return np.tanh(reservoir_state)  # Final activation

    def _update_intrinsic_plasticity(self, reservoir_state):
        """
        Update intrinsic plasticity parameters a (gain) and b (bias) following the original paper's formulation.
        """
        mu = self.target_mean
        sigma = np.sqrt(self.target_var)
        eta = self.ip_learning_rate

        # Flatten reservoir state for computation
        x = reservoir_state.ravel()  # Shape (reservoir_size,)

        # Compute bias update delta_b
        delta_b = -eta * ((-mu / (sigma ** 2)) + (x / (sigma ** 2)) + (1 - x ** 2) + mu * x)
        delta_b = delta_b.reshape(self.b.shape)  # Match shape (reservoir_size, 1)

        # Compute gain update delta_g
        x_net = np.dot(self.W, x.reshape(-1, 1))  # Net input to reservoir
        delta_g = eta / self.a + delta_b * x_net
        delta_g = delta_g.reshape(self.a.shape)  # Match shape (reservoir_size, 1)

        # Update gain and bias
        self.b += delta_b
        self.a += delta_g

        # Ensure numerical stability
        self.a = np.clip(self.a, 0.01, 10)
        self.b = np.clip(self.b, -5, 5)

    def fit(self, X_train, y_train, warm_up_steps=10, reservoir_state=None):
        n_samples = X_train.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))

        # Warm-up phase
        for t in range(warm_up_steps, n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)

        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            reservoir_states[t] = reservoir_state.ravel()
        extended_states = np.hstack([reservoir_states, X_train])

        # Check for NaN or Inf in extended states
        if not np.all(np.isfinite(extended_states)):
            raise ValueError("Non-finite values detected in extended states. Check IP or reservoir initialization.")

        self.W_out = np.dot(np.linalg.pinv(extended_states), y_train)

    def predict(self, X_test, reservoir_state=None):
        n_samples = X_test.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        predictions = np.zeros(n_samples)
        for t in range(n_samples):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            pred = np.dot(extended_state, self.W_out)
            predictions[t] = 1 if pred > 0.5 else 0  # Binary classification decision boundary at 0.5
        return predictions
