import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from time import time_ns
import matplotlib.pyplot as plt
from utils import calculate_f1_score, EchoStateNetworkWithIP

# File paths
training_data_location = 'occupancy_detection/datatraining.txt'
testing_data_location1 = 'occupancy_detection/datatest.txt'
testing_data_location2 = 'occupancy_detection/datatest2.txt'

# Read data
training_data = pd.read_csv(training_data_location, sep=',', header=0)
testing_data1 = pd.read_csv(testing_data_location1, sep=',', header=0)
testing_data2 = pd.read_csv(testing_data_location2, sep=',', header=0)

# Extract features and targets
features_train = training_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_train = training_data['Occupancy'].values  # Binary target
features_test1 = testing_data1[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test1 = testing_data1['Occupancy'].values
features_test2 = testing_data2[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test2 = testing_data2['Occupancy'].values

# Normalize features
scaler = StandardScaler()
features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test1 = scaler.transform(features_test1)
features_scaled_test2 = scaler.transform(features_test2)

# Sequentially split data to preserve time order
def split_data_time_ordered(features, targets, num_workers=3):
    """
    Split data sequentially to preserve time order for time series data.
    """
    split_size = len(features) // num_workers
    features_splits = [features[i * split_size:(i + 1) * split_size] for i in range(num_workers)]
    targets_splits = [targets[i * split_size:(i + 1) * split_size] for i in range(num_workers)]
    
    # Handle remainder by assigning leftover data to the last worker
    remainder = len(features) % num_workers
    if remainder > 0:
        features_splits[-1] = np.vstack([features_splits[-1], features[-remainder:]])
        targets_splits[-1] = np.hstack([targets_splits[-1], targets[-remainder:]])
    
    return features_splits, targets_splits

# Federated Averaging
def federated_averaging(weights_list):
    """
    Perform Federated Averaging on a list of weight matrices.
    """
    return np.mean(weights_list, axis=0)

# Prepare Federated Learning setup
num_workers = 3
features_splits, targets_splits = split_data_time_ordered(features_scaled_train, target_train, num_workers)

for i in range(num_workers):
    plt.figure(figsize=(14, 6))
    plt.plot(targets_splits[i], label='True', color='blue', linestyle='--')
    plt.title(f'Occupancy Detection - Worker {i + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Occupancy')
    plt.legend()
    plt.savefig(f'./FedNoDeep/worker_{i + 1}.png')

RC_size = 200
global_esn = EchoStateNetworkWithIP(input_size=5, reservoir_size=RC_size, output_size=1)

# Metrics storage
accuracies_test1, f1_scores_test1 = [], []
accuracies_test2, f1_scores_test2 = [], []
training_latencies, inference_latencies1, inference_latencies2 = [], [], []

# Federated Learning rounds
num_rounds = 10
for round_num in range(num_rounds):
    print(f"\n=== Round {round_num + 1} ===")
    
    # Step 1: Distribute global model to workers
    global_weights = global_esn.W_out if global_esn.W_out is not None else None
    
    local_weights = []
    local_training_latencies = []
    
    # Step 2: Local Training
    for worker_id in range(num_workers):
        print(f"Worker {worker_id + 1}: Training locally...")
        worker_esn = EchoStateNetworkWithIP(input_size=5, reservoir_size=RC_size, output_size=1)
        
        # Initialize with global weights
        if global_weights is not None:
            print(f"Worker {worker_id + 1}: Initializing with global weights...")
            worker_esn.W_out = global_weights
        
        # Train locally
        start_time = time_ns()
        worker_esn.fit(features_splits[worker_id], targets_splits[worker_id])
        train_time = (time_ns() - start_time) / 1e6  # ms
        local_training_latencies.append(train_time)

        worker_accuracy = accuracy_score(targets_splits[worker_id], worker_esn.predict(features_splits[worker_id]))
        print(f"Worker {worker_id + 1} Accuracy after training: {worker_accuracy}")
        
        print(f"Worker {worker_id + 1}: Training latency: {train_time:.2f} ms")
        local_weights.append(worker_esn.W_out)
    
    # Step 3: Aggregation at the central server
    print("Aggregating weights at the server...")
    aggregated_weights = federated_averaging(local_weights)
    global_esn.W_out = aggregated_weights  # Update global model
    
    # Step 4: Evaluate on the global model
    print("Evaluating global model on test datasets...")
    
    # Test on Dataset 1
    start_time = time_ns()
    y_pred_test1 = global_esn.predict(features_scaled_test1)
    inf_time_test1 = (time_ns() - start_time) / 1e6  # ms
    inference_latencies1.append(inf_time_test1)
    accuracy_test1 = accuracy_score(target_test1, y_pred_test1)
    f1_test1 = calculate_f1_score(target_test1, y_pred_test1)
    accuracies_test1.append(accuracy_test1)
    f1_scores_test1.append(f1_test1)
    print(f"Test 1 Accuracy: {accuracy_test1:.4f}, F1 Score: {f1_test1:.4f}")
    
    # Test on Dataset 2
    start_time = time_ns()
    y_pred_test2 = global_esn.predict(features_scaled_test2)
    inf_time_test2 = (time_ns() - start_time) / 1e6  # ms
    inference_latencies2.append(inf_time_test2)
    accuracy_test2 = accuracy_score(target_test2, y_pred_test2)
    f1_test2 = calculate_f1_score(target_test2, y_pred_test2)
    accuracies_test2.append(accuracy_test2)
    f1_scores_test2.append(f1_test2)
    print(f"Test 2 Accuracy: {accuracy_test2:.4f}, F1 Score: {f1_test2:.4f}")

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(range(1, num_rounds + 1), accuracies_test1, label='Test 1 Accuracy', marker='o')
plt.plot(range(1, num_rounds + 1), accuracies_test2, label='Test 2 Accuracy', marker='x')
plt.title('Federated Learning - Accuracy over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./FedNoDeep/federated_learning_accuracy.png')

plt.figure(figsize=(14, 6))
plt.plot(range(1, num_rounds + 1), f1_scores_test1, label='Test 1 F1 Score', marker='o')
plt.plot(range(1, num_rounds + 1), f1_scores_test2, label='Test 2 F1 Score', marker='x')
plt.title('Federated Learning - F1 Score over Rounds')
plt.xlabel('Round')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('./FedNoDeep/federated_learning_f1_scores.png')

# plot last round results
plt.figure(figsize=(14, 6))
plt.plot(target_test1, label='True', color='blue', linestyle='--')
plt.plot(y_pred_test1, label='Predicted', color='red', linestyle='-')
plt.title(f'Occupancy Detection - Test 1 Accuracy: {accuracy_test1:.4f}')
plt.xlabel('Time Steps')
plt.ylabel('Occupancy')
plt.legend()
plt.savefig('./FedNoDeep/federated_learning_test1.png')

plt.figure(figsize=(14, 6))
plt.plot(target_test2, label='True', color='blue', linestyle='--')
plt.plot(y_pred_test2, label='Predicted', color='red', linestyle='-')
plt.title(f'Occupancy Detection - Test 2 Accuracy: {accuracy_test2:.4f}')
plt.xlabel('Time Steps')
plt.ylabel('Occupancy')
plt.legend()
plt.savefig('./FedNoDeep/federated_learning_test2.png')