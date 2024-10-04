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
