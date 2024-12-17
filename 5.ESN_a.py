import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time_ns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from utils import calculate_f1_score, EchoStateNetwork

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

class EchoStateNetworkWithSoftmax:
    def __init__(self, input_size, reservoir_size, output_size, input_scaling=0.1, spectral_radius=0.2, sparsity=0.3, random_seed=42, target_mean=0.2, target_var=0.1, ip_learning_rate=0.1):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size)) * input_scaling
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask
        spectral_radius_current = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / spectral_radius_current) * spectral_radius
        self.W_out = None
        self.reservoir_state = np.zeros((reservoir_size, 1))
        self.encoder = OneHotEncoder(sparse_output=False, categories='auto')

        # Intrinsic plasticity parameters
        self.a = np.ones((reservoir_size, 1))  # Gain
        self.b = np.zeros((reservoir_size, 1))  # Biases
        self.target_mean = target_mean
        self.target_var = target_var
        self.ip_learning_rate = ip_learning_rate  # Adjustable learning rate for IP

    # def _update_reservoir(self, X, prev_reservoir_state):
    #     X = X.reshape(-1, 1)
    #     prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
    #     pre_activation = np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)
    #     reservoir_state = np.tanh(pre_activation)
    #     return reservoir_state

    def _update_reservoir(self, X, prev_reservoir_state):

        X = X.reshape(-1, 1)
        prev_reservoir_state = prev_reservoir_state.reshape(-1, 1)
        pre_activation = self.a * (np.dot(self.W_in, X) + np.dot(self.W, prev_reservoir_state)) + self.b

        # Apply intrinsic plasticity adaptation
        reservoir_state = np.tanh(pre_activation)
        # reservoir_state = self.a * reservoir_state + self.b
        self._update_intrinsic_plasticity(reservoir_state)

        # Clip reservoir state to prevent overflow
        # reservoir_state = np.clip(reservoir_state, -5, 5)
        return np.tanh(reservoir_state)  # Final activation

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
        # delta_g = eta / self.a + delta_b * x_net
        delta_g = eta / self.a + delta_b * reservoir_state
        delta_g = delta_g.reshape(self.a.shape)  # Match shape (200, 1)

        # Update gain and bias
        self.b += delta_b
        self.a += delta_g

        # Ensure numerical stability
        self.a = np.clip(self.a, 0.01, 10)
        self.b = np.clip(self.b, -5, 5)

    def fit(self, X_train, y_train):
        """
        Fit the ESN using Ridge Regression.
        Parameters:
        - X_train: Input features (n_samples, n_features)
        - y_train: Target labels (n_samples,)
        """
        n_samples = X_train.shape[0]
        reservoir_states = []

        # Collect reservoir states for the training data
        for t in range(n_samples):
            self.reservoir_state = self._update_reservoir(X_train[t], self.reservoir_state)
            reservoir_states.append(self.reservoir_state.ravel())

        reservoir_states = np.array(reservoir_states)

        # One-hot encode the target labels for multi-class classification
        y_train_one_hot = self.encoder.fit_transform(y_train.reshape(-1, 1))

        # Train the readout layer using Ridge regression on reservoir states only
        ridge = Ridge(alpha=1e-4)  # Regularization parameter
        ridge.fit(reservoir_states, y_train_one_hot)
        self.W_out = ridge.coef_.T  # Save weights
    
    def predict(self, X_test, target_test=None):
        """
        Predict class probabilities and labels for test data.
        Parameters:
        - X_test: Input features (n_samples, n_features)
        - target_test: True target labels (optional, for visualization)
        Returns:
        - predictions: Predicted class labels (n_samples,)
        - probabilities: Predicted class probabilities (n_samples, n_classes)
        """
        n_samples = X_test.shape[0]
        reservoir_states = []

        # Collect reservoir states for the test data
        for t in range(n_samples):
            self.reservoir_state = self._update_reservoir(X_test[t], self.reservoir_state)
            reservoir_states.append(self.reservoir_state.ravel())

        reservoir_states = np.array(reservoir_states)

        # Compute logits using only the reservoir states
        logits = np.dot(reservoir_states, self.W_out)
        
        # Apply softmax to compute probabilities
        probabilities = softmax(logits, axis=1)

        # Predict the class with the highest probability
        predictions = np.argmax(probabilities, axis=1)

        # Optional: Plot predictions vs. target if target_test is provided
        if target_test is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(target_test, label='True Labels', alpha=0.7, color='blue')
            plt.plot(predictions, label='Predicted Labels', alpha=0.7, color='orange')
            plt.title("Predictions vs. True Labels")
            plt.legend()
            plt.show()

        return predictions, probabilities


    def visualize_extended_states_tsne(self, X_test, target_test):
        # Collect extended states
        extended_states = []
        reservoir_states = []
        
        # Reset reservoir state
        reservoir_state = np.zeros((self.reservoir_size, 1))
        
        # Collect extended states for all samples
        for t in range(X_test.shape[0]):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            
            # Collect extended state and reservoir state
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            extended_states.append(extended_state)
            reservoir_states.append(reservoir_state.ravel())
        
        # Convert to numpy arrays
        extended_states = np.array(extended_states)
        reservoir_states = np.array(reservoir_states)
        
        # Perform t-SNE on extended states
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        extended_states_2d = tsne.fit_transform(extended_states)
        
        # Optional: Separate visualizations
        plt.figure(figsize=(16, 5))
        
        # Subplot 1: Extended States t-SNE
        plt.subplot(131)
        scatter = plt.scatter(
            extended_states_2d[:, 0], 
            extended_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Extended States t-SNE')
        plt.colorbar(scatter)
        
        # Subplot 2: Raw Reservoir States t-SNE
        reservoir_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reservoir_states_2d = reservoir_tsne.fit_transform(reservoir_states)
        
        plt.subplot(132)
        scatter = plt.scatter(
            reservoir_states_2d[:, 0], 
            reservoir_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Reservoir States t-SNE')
        plt.colorbar(scatter)
        
        # Subplot 3: Input Features t-SNE
        input_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        input_states_2d = input_tsne.fit_transform(X_test)
        
        plt.subplot(133)
        scatter = plt.scatter(
            input_states_2d[:, 0], 
            input_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Input Features t-SNE')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        return extended_states_2d

    # Additional diagnostic function
    def compute_state_separability(extended_states, target_test):
        """
        Compute metrics to assess how well extended states separate classes
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        # Silhouette Score (higher is better)
        silhouette = silhouette_score(extended_states, target_test)
        
        # Davies-Bouldin Index (lower is better)
        davies_bouldin = davies_bouldin_score(extended_states, target_test)
        
        print("\nClass Separability Metrics:")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return silhouette, davies_bouldin


class EchoStateNetworkWithIP:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, random_seed=42, target_mean=0.2, target_var=0.1, ip_learning_rate=0.6):
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
        self.a = np.ones((reservoir_size, 1))  # Gain
        self.b = np.zeros((reservoir_size, 1))  # Biases
        self.target_mean = target_mean
        self.target_var = target_var
        self.ip_learning_rate = ip_learning_rate  # Adjustable learning rate for IP

    # with intrinsic plasticity    
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

    def fit(self, X_train, y_train, warm_up_steps=100, forgetting_factor=0.99, ridge_param=1e-4):
        """
        Online learning with Ridge regression using Recursive Least Squares
        
        Parameters:
        - X_train: Input features
        - y_train: Target values
        - warm_up_steps: Initial steps to warm up the reservoir
        - forgetting_factor: Controls the importance of past observations (0.95-0.999)
        - ridge_param: Regularization parameter for Ridge regression
        """
        n_samples = X_train.shape[0]
        reservoir_state = np.zeros((self.reservoir_size, 1))

        # Initialize covariance matrix and weights
        extended_state_size = self.reservoir_size + self.input_size
        self.P = np.eye(extended_state_size) / ridge_param  # Initial covariance matrix
        self.W_out = np.zeros((extended_state_size, self.output_size))

        for t in range(warm_up_steps, n_samples):
            X = X_train[t]
            
            # Update reservoir state
            reservoir_state = self._update_reservoir(X, reservoir_state)
            
            # Create extended state
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            extended_state = extended_state.reshape(-1, 1)
            
            # Target value
            y = y_train[t]
            
            # Compute Kalman gain
            numerator = self.P @ extended_state
            denominator = (forgetting_factor + extended_state.T @ self.P @ extended_state)[0, 0]
            K = numerator / denominator

            # Prediction error
            error = y - extended_state.T @ self.W_out

            # Update output weights
            self.W_out += K @ error

            # Update covariance matrix
            self.P = (1 / forgetting_factor) * (
                self.P - 
                (K @ extended_state.T @ self.P)
            )

        return self
    
    def predict(self, X_test, target_test, reservoir_state=None):
        n_samples = X_test.shape[0]
        if reservoir_state is None:
            reservoir_state = np.zeros((self.reservoir_size, 1))
        predictions = np.zeros(n_samples)
        raw_prediction = []
        for t in range(n_samples):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            pred = np.dot(extended_state, self.W_out)
            raw_prediction += [pred]
            predictions[t] = 1 if pred > 0.5 else 0  # Binary classification decision boundary at 0.5

        # plot targets and 
        plt.plot(target_test)
        plt.plot(raw_prediction)
        plt.show()
        
        return predictions, None

    def visualize_extended_states_tsne(self, X_test, target_test):
        # Collect extended states
        extended_states = []
        reservoir_states = []
        
        # Reset reservoir state
        reservoir_state = np.zeros((self.reservoir_size, 1))
        
        # Collect extended states for all samples
        for t in range(X_test.shape[0]):
            X = X_test[t]
            reservoir_state = self._update_reservoir(X, reservoir_state)
            
            # Collect extended state and reservoir state
            extended_state = np.hstack([reservoir_state.ravel(), X.ravel()])
            extended_states.append(extended_state)
            reservoir_states.append(reservoir_state.ravel())
        
        # Convert to numpy arrays
        extended_states = np.array(extended_states)
        reservoir_states = np.array(reservoir_states)
        
        # Perform t-SNE on extended states
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        extended_states_2d = tsne.fit_transform(extended_states)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with different colors for each class
        scatter = plt.scatter(
            extended_states_2d[:, 0], 
            extended_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        
        plt.colorbar(scatter, label='Class')
        plt.title('t-SNE Visualization of Extended States')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.show()
        
        # Optional: Separate visualizations
        plt.figure(figsize=(16, 5))
        
        # Subplot 1: Extended States t-SNE
        plt.subplot(131)
        scatter = plt.scatter(
            extended_states_2d[:, 0], 
            extended_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Extended States t-SNE')
        plt.colorbar(scatter)
        
        # Subplot 2: Raw Reservoir States t-SNE
        reservoir_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reservoir_states_2d = reservoir_tsne.fit_transform(reservoir_states)
        
        plt.subplot(132)
        scatter = plt.scatter(
            reservoir_states_2d[:, 0], 
            reservoir_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Reservoir States t-SNE')
        plt.colorbar(scatter)
        
        # Subplot 3: Input Features t-SNE
        input_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        input_states_2d = input_tsne.fit_transform(X_test)
        
        plt.subplot(133)
        scatter = plt.scatter(
            input_states_2d[:, 0], 
            input_states_2d[:, 1], 
            c=target_test, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Input Features t-SNE')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        return extended_states_2d

    # Additional diagnostic function
    def compute_state_separability(extended_states, target_test):
        """
        Compute metrics to assess how well extended states separate classes
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        # Silhouette Score (higher is better)
        silhouette = silhouette_score(extended_states, target_test)
        
        # Davies-Bouldin Index (lower is better)
        davies_bouldin = davies_bouldin_score(extended_states, target_test)
        
        print("\nClass Separability Metrics:")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return silhouette, davies_bouldin


def run_experiment(esn, run_with_ip, features_train, target_train, features_test1, target_test1, features_test2, target_test2, RC_size, accuracies_test1, f1_scores_test1, accuracies_test2, f1_scores_test2, training_latencies, inference_latencies1, inference_latencies2):
    print(f'Running ESN with reservoir size {RC_size}')
    print("-----------------------------------------")

    # esn = EchoStateNetwork(input_size=5, reservoir_size=RC_size, output_size=1)  # Adjust reservoir size as needed
    # esn = EchoStateNetworkWithIP(input_size=5, reservoir_size=RC_size, output_size=1)  # Adjust reservoir size as needed
    start_time = time_ns()
    esn.fit(features_train, target_train)
    # esn.old_fit(features_train, target_train)
    train_time = (time_ns() - start_time)
    training_latencies.append(train_time)
    print(f'Training time for reservoir size {RC_size}: {train_time / 1e6:.2f} ms')

    # Step 3: Train/test split (keep time dependency in mind)
    # X_train, X_test, y_train, y_test = features_train, features_test1, target_train, target_test1
    # Step 5: Train and test the ESN

    start_time = time_ns()
    # y_pred_1 = esn.predict(features_test1, target_test1)
    y_pred_1, _ = esn.predict(features_test1, target_test1)
    esn.visualize_extended_states_tsne(features_test1, target_test1)
    inf_time = (time_ns() - start_time) 
    inference_latencies1.append(inf_time)

    # Step 6: Evaluate the performance
    accuracy = accuracy_score(target_test1, y_pred_1)

    f1 = calculate_f1_score(target_test1, y_pred_1)
    print(f"a) RC_size{RC_size} accuracy: {accuracy:.4f} f1: {f1:.4f}")   
    accuracies_test1.append(accuracy)
    f1_scores_test1.append(f1)

    folder = "ESN_results" if not run_with_ip else "ESN_results_withIP"

    start_time = time_ns()
    # y_pred_2 = esn.predict(features_test2, target_test2)
    y_pred_2, _ = esn.predict(features_test2, target_test2)
    esn.visualize_extended_states_tsne(features_test2, target_test2)
    inf_time = (time_ns() - start_time)
    inference_latencies2.append(inf_time)

    accuracy = accuracy_score(target_test2, y_pred_2)
    f1 = calculate_f1_score(target_test2, y_pred_2)
    print(f"b) RC_size{RC_size} accuracy: {accuracy:.4f} f1: {f1:.4f}")   
    accuracies_test2.append(accuracy)
    f1_scores_test2.append(f1)

training_data_location = 'occupancy_detection/datatraining.txt'
testing_data_location1 = 'occupancy_detection/datatest.txt'
testing_data_location2 = 'occupancy_detection/datatest2.txt'

# read data from txt file datatrainin.txt
training_data = pd.read_csv(training_data_location, sep=',', header=0)
testing_data1 = pd.read_csv(testing_data_location1, sep=',', header=0)
testing_data2 = pd.read_csv(testing_data_location2, sep=',', header=0)

# Step 1: Extract features and target
features_train = training_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_train = training_data['Occupancy'].values  # True/False or 1/0 for occupancy

features_test1 = testing_data1[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test1 = testing_data1['Occupancy'].values  # True/False or 1/0 for occupancy

features_test2 = testing_data2[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test2 = testing_data2['Occupancy'].values  # True/False or 1/0 for occupancy

# Step 2: Normalize features
scaler = StandardScaler()
features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test1 = scaler.transform(features_test1)
features_scaled_test2 = scaler.transform(features_test2)

# explore RC sizes logaritmically
RC_sizes = [50]
accuracies_test1 = []
f1_scores_test1 = []
accuracies_test2 = []
f1_scores_test2 = []

training_latencies = []
inference_latencies1 = []
inference_latencies2 = []

run_with_ip = True
result_folder = 'ESN_results' if not run_with_ip else 'ESN_results_withIP'

for RC_size in RC_sizes:
    esn = EchoStateNetwork(input_size=5, reservoir_size=RC_size, output_size=1)
    if run_with_ip:
        esn = EchoStateNetworkWithSoftmax(input_size=5, reservoir_size=RC_size, output_size=1)
    run_experiment(esn, run_with_ip, features_scaled_train, target_train, features_scaled_test1, target_test1, features_scaled_test2, target_test2, RC_size, accuracies_test1, f1_scores_test1, accuracies_test2, f1_scores_test2, training_latencies, inference_latencies1, inference_latencies2)
    
for index, RC_size in enumerate(RC_sizes):
    print("Size: ", RC_size)
    print("f1a", f1_scores_test1[index])
    print("f1b", f1_scores_test2[index])
    print("-----------------------------")
    print("accuracya", accuracies_test1[index])
    print("accuracyb", accuracies_test2[index])
    print("-----------------------------")
    print("-----------------------------")