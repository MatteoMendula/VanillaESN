import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from time import time_ns
from utils import calculate_f1_score, EchoStateNetworkWithIP_FedIp

# Data Loading and Preparation
training_data_location = 'occupancy_detection/datatraining.txt'
testing_data_location1 = 'occupancy_detection/datatest.txt'
testing_data_location2 = 'occupancy_detection/datatest2.txt'

# Load data
training_data = pd.read_csv(training_data_location, sep=',', header=0)
testing_data1 = pd.read_csv(testing_data_location1, sep=',', header=0)
testing_data2 = pd.read_csv(testing_data_location2, sep=',', header=0)

# Extract features and target
features_train = training_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_train = training_data['Occupancy'].values

features_test1 = testing_data1[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test1 = testing_data1['Occupancy'].values

features_test2 = testing_data2[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test2 = testing_data2['Occupancy'].values

# Normalize features
scaler = StandardScaler()
features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test1 = scaler.transform(features_test1)
features_scaled_test2 = scaler.transform(features_test2)

# Split data for 3 workers while keeping order
split_size = len(features_scaled_train) // 3
worker_data = [
    (features_scaled_train[i * split_size:(i + 1) * split_size],
     target_train[i * split_size:(i + 1) * split_size])
    for i in range(3)
]

# Hyperparameters
RC_SIZE = 100
N_ROUNDS = 10

# Tracking metrics
accuracy_test1 = []
accuracy_test2 = []

# Initialize workers
workers = [EchoStateNetworkWithIP_FedIp(input_size=5, reservoir_size=RC_SIZE, output_size=1) for _ in range(3)]

# Initialize global gain and bias
global_gain = np.zeros(RC_SIZE)
global_bias = np.zeros(RC_SIZE)

# Federated Training Loop
for round_idx in range(1, N_ROUNDS + 1):
    print(f"--- Round {round_idx} ---")

    # Local updates
    local_gain_updates = []
    local_bias_updates = []

    for worker_idx, worker in enumerate(workers):
        print(f"Worker {worker_idx + 1} training...")

        # Set the global gain and bias to each worker
        # add one dimension to global_gain and global_bias
        if global_gain.ndim == 1 and global_bias.ndim == 1:
            global_gain = global_gain.reshape(-1, 1)
            global_bias = global_bias.reshape(-1, 1)
        worker.set_ip_parameters(global_gain, global_bias)

        # Train the worker
        features, target = worker_data[worker_idx]
        worker.fit(features, target)

        # Collect local gain and bias
        gain, bias = worker.get_ip_parameters()
        local_gain_updates.append(gain)
        local_bias_updates.append(bias)

    # Aggregate the gain and bias using FedAvg
    global_gain = np.mean(local_gain_updates, axis=0)
    global_bias = np.mean(local_bias_updates, axis=0)

    # Evaluate on Test 1
    print("Evaluating on Test 1...")
    predictions_test1 = workers[0].predict(features_scaled_test1)  # All workers are identical after aggregation
    acc_test1 = accuracy_score(target_test1, predictions_test1)
    accuracy_test1.append(acc_test1)
    print(f"Round {round_idx} Test 1 Accuracy: {acc_test1:.4f}")

    # Evaluate on Test 2
    print("Evaluating on Test 2...")
    predictions_test2 = workers[0].predict(features_scaled_test2)
    acc_test2 = accuracy_score(target_test2, predictions_test2)
    accuracy_test2.append(acc_test2)
    print(f"Round {round_idx} Test 2 Accuracy: {acc_test2:.4f}")

# Plot Accuracy Over Rounds
plt.figure(figsize=(10, 6))
plt.plot(range(1, N_ROUNDS + 1), accuracy_test1, label='Test 1 Accuracy', marker='o')
plt.plot(range(1, N_ROUNDS + 1), accuracy_test2, label='Test 2 Accuracy', marker='x')
plt.title('Federated Learning - Accuracy over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('./FedIp/federated_learning_accuracy.png')
plt.show()

