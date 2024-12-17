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


class WESAD_EchoStateNetworkWithIP_FedIp:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, random_seed=42, target_mean=0.2, target_var=0.1, ip_learning_rate=0.5):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size  # Number of classes for multi-class classification
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size)) * 0.1
        
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask
        spectral_radius_current = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / spectral_radius_current) * spectral_radius
        
        self.W_out = np.random.uniform(-1, 1, (reservoir_size + input_size, output_size)) * 0.1

        # Intrinsic plasticity parameters
        self.a = np.ones((reservoir_size, 1))
        self.b = np.zeros((reservoir_size, 1))
        self.target_mean = target_mean
        self.target_var = target_var
        self.ip_learning_rate = ip_learning_rate

    def set_ip_parameters(self, a, b):
        if a.shape != self.a.shape or b.shape != self.b.shape:
            raise ValueError("Shape mismatch: a and b must match the reservoir size.", a.shape, self.a.shape, b.shape, self.b.shape)
        self.a = np.copy(a)
        self.b = np.copy(b)

    def get_ip_parameters(self):
        return np.copy(self.a), np.copy(self.b)
    
    def set_W_out(self, W_out):
        """
        Set the readout weights (W_out).
        Args:
            W_out (numpy.ndarray): Readout weights.
        """
        if W_out.shape != (self.reservoir_size + self.input_size, self.output_size):
            print(f"Shape mismatch: W_out must be of shape {(self.reservoir_size + self.input_size, self.output_size)}")
            print(f"Received shape: {W_out.shape}")
            raise ValueError("Shape mismatch for W_out.")
        self.W_out = np.copy(W_out)

    def get_W_out(self):
        """
        Get the readout weights (W_out).
        Returns:
            numpy.ndarray: Readout weights.
        """
        return np.copy(self.W_out)

    def _update_reservoir(self, X, prev_reservoir_state):
        X = X.reshape(-1, 1)
        prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
        pre_activation = np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)

        reservoir_state = np.tanh(pre_activation)
        reservoir_state = self.a * reservoir_state + self.b
        self._update_intrinsic_plasticity(reservoir_state)

        return np.tanh(np.clip(reservoir_state, -5, 5))

    def _update_intrinsic_plasticity(self, reservoir_state):
        mu = self.target_mean
        sigma = np.sqrt(self.target_var)
        eta = self.ip_learning_rate

        x = reservoir_state.ravel()

        delta_b = -eta * ((-mu / (sigma ** 2)) + (x / (sigma ** 2)) + (1 - x ** 2) + mu * x)
        delta_b = delta_b.reshape(self.b.shape)

        x_net = np.dot(self.W, x.reshape(-1, 1))
        
        # Prevent division by zero or very small values
        safe_a = np.clip(self.a, 1e-6, None)
        
        delta_g = eta / safe_a + delta_b * x_net
        delta_g = delta_g.reshape(self.a.shape)

        self.b += delta_b
        self.a += delta_g

        # Ensure stability
        self.a = np.clip(self.a, 0.01, 10)
        self.b = np.clip(self.b, -5, 5)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)


    def check_wout_shape(self, y_train):
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        if n_classes != self.output_size:
            self.output_size = n_classes
            self.W_out = np.random.uniform(-1, 1, (self.reservoir_size + self.input_size, self.output_size))
            print(f"Reinitialized W_out with shape {self.W_out.shape}")

    def fit(self, X_train, y_train, warm_up_steps=1000, reservoir_state=None):
        n_samples = X_train.shape[0]
        
        # Dynamically determine the number of classes
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        # Reinitialize W_out if class count changes or is different from current output size
        if n_classes != self.output_size:
            self.output_size = n_classes
            self.W_out = np.random.uniform(-1, 1, (self.reservoir_size + self.input_size, self.output_size))
            print(f"Reinitialized W_out with shape {self.W_out.shape}")

        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        
        # Warm up phase
        for t in range(0, warm_up_steps):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)

        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            reservoir_states[t] = reservoir_state.ravel()

        extended_states = np.hstack([reservoir_states, X_train])

        # Label encoding for binary or multi-class
        if n_classes == 2:
            y_encoded = y_train.reshape(-1, 1)  # Binary case
        else:
            y_encoded = np.zeros((y_train.size, self.output_size))  # Multi-class case
            y_encoded[np.arange(y_train.size), y_train] = 1
        
        reg = 1e-6
        self.W_out = np.dot(np.linalg.pinv(extended_states.T @ extended_states + reg * np.eye(extended_states.shape[1])) @ extended_states.T, y_encoded)

    def fit_online(self, X_train, y_train, warm_up_steps=1000, reservoir_state=None, lambda_reg=1e-6):
        """
        Online training using Recursive Least Squares (RLS) for a reservoir computing model.

        Parameters:
            X_train (ndarray): Input data of shape (n_samples, input_size).
            y_train (ndarray): Target labels of shape (n_samples,).
            warm_up_steps (int): Number of initial steps to stabilize the reservoir state.
            reservoir_state (ndarray): Initial reservoir state, optional.
            lambda_reg (float): Regularization parameter for RLS.
        """
        n_samples = X_train.shape[0]
        unique_classes = np.unique(y_train)

        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))

        # Warm-up phase to stabilize the reservoir
        for t in range(min(warm_up_steps, n_samples)):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)

        # Initialize RLS matrices
        P = np.eye(self.reservoir_size + self.input_size) / lambda_reg

        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            
            # Construct extended state [reservoir_state; input]
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()]).reshape(-1, 1)
            
            # Encode the target as a one-hot vector
            target = np.zeros((self.output_size, 1))
            class_index = np.where(unique_classes == y_train[t])[0][0]  # Map y_train to class index
            target[class_index] = 1
            
            # Compute error and update RLS weights
            error = target - np.dot(self.W_out.T, extended_state)
            gain = P @ extended_state / (1 + extended_state.T @ P @ extended_state)
            self.W_out += gain @ error.T
            P -= gain @ extended_state.T @ P

    # # This version updates the output weights incrementally using the Recursive Least Squares (RLS) algorithm
    # def fit_online(self, X_train, y_train, warm_up_steps=1000, reservoir_state=None, lambda_reg=1e-6):
    #     n_samples = X_train.shape[0]
        
    #     unique_classes = np.unique(y_train)
    #     n_classes = len(unique_classes)
        
    #     # Reinitialize W_out if class count changes
    #     if n_classes != self.output_size:
    #         self.output_size = n_classes
    #         self.W_out = np.random.uniform(-1, 1, (self.reservoir_size + self.input_size, n_classes))
        
    #     if reservoir_state is None:
    #         reservoir_state = np.zeros((self.reservoir_size, 1))
        
    #     # Warm-up phase
    #     for t in range(warm_up_steps):
    #         X = X_train[t]
    #         reservoir_state = self._update_reservoir(X, reservoir_state)
        
    #     # Initialize RLS matrices
    #     P = np.eye(self.reservoir_size + self.input_size) / lambda_reg

    #     for t in range(n_samples):
    #         X = X_train[t]
    #         reservoir_state = self._update_reservoir(X, reservoir_state)
    #         extended_state = np.hstack([reservoir_state.ravel(), X.ravel()]).reshape(-1, 1)
            
    #         # Encode label for binary or multi-class
    #         target = np.zeros((self.output_size, 1))
    #         if self.output_size == 2:
    #             target[0] = y_train[t]  # Binary case
    #         else:
    #             target[y_train[t]] = 1  # Multi-class case

    #         error = target - np.dot(self.W_out.T, extended_state)
    #         gain = P @ extended_state / (1 + extended_state.T @ P @ extended_state)
    #         self.W_out += gain @ error.T
    #         print("shape of wout", self.W_out.shape)
    #         P -= gain @ extended_state.T @ P

    # This method processes a sliding window of reservoir states instead of storing all states
    def fit_rolling_window(self, X_train, y_train, warm_up_steps=1000, window_size=100, reservoir_state=None):
        n_samples = X_train.shape[0]
        
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        if n_classes != self.output_size:
            self.output_size = n_classes
            self.W_out = np.random.uniform(-1, 1, (self.reservoir_size + self.input_size, n_classes))
        
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        
        for t in range(warm_up_steps):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
        
        rolling_states = np.zeros((window_size, self.reservoir_size + self.input_size))
        y_encoded_window = np.zeros((window_size, self.output_size))
        
        for t in range(n_samples):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            rolling_states[t % window_size] = extended_state
            
            # Update label encoding for rolling window
            y_encoded = np.zeros(self.output_size)
            if self.output_size == 2:
                y_encoded[0] = y_train[t]
            else:
                y_encoded[y_train[t]] = 1
            
            y_encoded_window[t % window_size] = y_encoded
            
            if (t + 1) % window_size == 0:
                reg = 1e-6
                self.W_out = np.linalg.pinv(rolling_states.T @ rolling_states + reg * np.eye(rolling_states.shape[1])) @ rolling_states.T @ y_encoded_window

    # This approach processes the data in batches rather than storing the entire dataset
    def fit_mini_batch(self, X_train, y_train, batch_size=100, warm_up_steps=1000, reservoir_state=None):
        n_samples = X_train.shape[0]
        
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        if n_classes != self.output_size:
            self.output_size = n_classes
            self.W_out = np.random.uniform(-1, 1, (self.reservoir_size + self.input_size, n_classes))
        
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        
        for t in range(warm_up_steps):
            X = X_train[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X_train[start:end]
            batch_y = y_train[start:end]
            
            reservoir_states = []
            for X in batch_X:
                reservoir_state = self._update_reservoir(X, reservoir_state)
                reservoir_states.append(np.hstack([reservoir_state.ravel(), X.ravel()]))
            
            reservoir_states = np.array(reservoir_states)
            y_encoded = np.zeros((batch_y.size, self.output_size))
            for i, label in enumerate(batch_y):
                if self.output_size == 2:
                    y_encoded[i, 0] = label  # Binary case
                else:
                    y_encoded[i, label] = 1  # Multi-class
            
            reg = 1e-6
            self.W_out = np.linalg.pinv(reservoir_states.T @ reservoir_states + reg * np.eye(reservoir_states.shape[1])) @ reservoir_states.T @ y_encoded


    def predict(self, X_test, reservoir_state=None):
        n_samples = X_test.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))

        predictions = np.zeros(n_samples)
        for t in range(n_samples):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            output = np.dot(extended_state, self.W_out)
            
            if self.output_size == 1:
                probability = self.sigmoid(output)
                predictions[t] = 1 if probability > 0.5 else 0  # Binary case
            else:
                probabilities = self.softmax(output.reshape(1, -1))
                predictions[t] = np.argmax(probabilities)  # Multi-class case
        return predictions