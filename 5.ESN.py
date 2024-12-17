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

from utils import calculate_f1_score, EchoStateNetwork, EchoStateNetworkWithIP

training_data_location = 'occupancy_detection/datatraining.txt'
testing_data_location1 = 'occupancy_detection/datatest.txt'
testing_data_location2 = 'occupancy_detection/datatest2.txt'

def run_experiment(esn, run_with_ip, features_train, target_train, features_test1, target_test1, features_test2, target_test2, RC_size, accuracies_test1, f1_scores_test1, accuracies_test2, f1_scores_test2, training_latencies, inference_latencies1, inference_latencies2):
    print(f'Running ESN with reservoir size {RC_size}')
    print("-----------------------------------------")

    # esn = EchoStateNetwork(input_size=5, reservoir_size=RC_size, output_size=1)  # Adjust reservoir size as needed
    # esn = EchoStateNetworkWithIP(input_size=5, reservoir_size=RC_size, output_size=1)  # Adjust reservoir size as needed
    start_time = time_ns()
    esn.fit_online(features_train, target_train)
    # esn.old_fit(features_train, target_train)
    train_time = (time_ns() - start_time)
    training_latencies.append(train_time)
    print(f'Training time for reservoir size {RC_size}: {train_time / 1e6:.2f} ms')

    # Step 3: Train/test split (keep time dependency in mind)
    # X_train, X_test, y_train, y_test = features_train, features_test1, target_train, target_test1
    # Step 5: Train and test the ESN

    start_time = time_ns()
    y_pred_1 = esn.old_predict(features_test1)
    inf_time = (time_ns() - start_time) 
    inference_latencies1.append(inf_time)

    # Step 6: Evaluate the performance
    accuracy = accuracy_score(target_test1, y_pred_1)
    f1 = calculate_f1_score(target_test1, y_pred_1)
    accuracies_test1.append(accuracy)
    f1_scores_test1.append(f1)

    folder = "ESN_results" if not run_with_ip else "ESN_results_withIP"

    # Step 7: Visualize the results
    plt.figure(figsize=(14, 6))
    plt.plot(target_test1, label='True', color='blue', linestyle='--')
    plt.plot(y_pred_1, label='Predicted', color='red', linestyle='-')
    plt.title(f'Occupancy Detection - Test Accuracy: {accuracy:.4f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Occupancy')
    plt.legend()
    plt.savefig(f'./{folder}/test1/occupancy_detection_esn_test1_size{RC_size}.png')

    start_time = time_ns()
    y_pred_2 = esn.old_predict(features_test2)
    inf_time = (time_ns() - start_time)
    inference_latencies2.append(inf_time)

    accuracy = accuracy_score(target_test2, y_pred_2)
    f1 = calculate_f1_score(target_test2, y_pred_2)
    accuracies_test2.append(accuracy)
    f1_scores_test2.append(f1)

    plt.figure(figsize=(14, 6))
    plt.plot(target_test2, label='True', color='blue', linestyle='--')
    plt.plot(y_pred_2, label='Predicted', color='red', linestyle='-')
    plt.title(f'Occupancy Detection - Test Accuracy: {accuracy:.4f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Occupancy')
    plt.legend()
    plt.savefig(f'./{folder}/test2/occupancy_detection_esn_test2_size{RC_size}.png')

# read data from txt file datatrainin.txt
training_data = pd.read_csv(training_data_location, sep=',', header=0)
testing_data1 = pd.read_csv(testing_data_location1, sep=',', header=0)
testing_data2 = pd.read_csv(testing_data_location2, sep=',', header=0)

# Step 1: Extract features and target
features_train = training_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_train = training_data['Occupancy'].values  # True/False or 1/0 for occupancy

print(features_train.shape)
print(target_train.shape)

features_test1 = testing_data1[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test1 = testing_data1['Occupancy'].values  # True/False or 1/0 for occupancy

features_test2 = testing_data2[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test2 = testing_data2['Occupancy'].values  # True/False or 1/0 for occupancy

# Step 2: Normalize features
scaler = StandardScaler()
features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test1 = scaler.transform(features_test1)
features_scaled_test2 = scaler.transform(features_test2)


RC_sizes = [50, 100, 200, 300, 400]
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
        esn = EchoStateNetworkWithIP(input_size=5, reservoir_size=RC_size, output_size=1)
    run_experiment(esn, run_with_ip, features_scaled_train, target_train, features_scaled_test1, target_test1, features_scaled_test2, target_test2, RC_size, accuracies_test1, f1_scores_test1, accuracies_test2, f1_scores_test2, training_latencies, inference_latencies1, inference_latencies2)
    
for index, RC_size in enumerate(RC_sizes):
    print("Size: ", RC_size)
    print("f1a", f1_scores_test1[index])
    print("f1b", f1_scores_test2[index])
    print("-----------------------------")


# Step 8: Visualize the results for both datasets
plt.figure(figsize=(14, 6))
plt.plot(RC_sizes, accuracies_test1, label='Test 1 Accuracy', color='blue', marker='o')
plt.plot(RC_sizes, accuracies_test2, label='Test 2 Accuracy', color='red', marker='x')
plt.title('Occupancy Detection - ESN Accuracy - run_with_ip: ' + str(run_with_ip))
plt.xlabel('Reservoir Size')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'./{result_folder}/occupancy_detection_esn_accuracies.png')

plt.figure(figsize=(14, 6))
plt.plot(RC_sizes, f1_scores_test1, label='Test 1 F1 Score', color='blue', marker='o')
plt.plot(RC_sizes, f1_scores_test2, label='Test 2 F1 Score', color='red', marker='x')
plt.title('Occupancy Detection - ESN F1 Score - run_with_ip: ' + str(run_with_ip))
plt.xlabel('Reservoir Size')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig(f'./{result_folder}/occupancy_detection_esn_f1_scores.png')

plt.figure(figsize=(14, 6))
plt.plot(RC_sizes, training_latencies, label='Training Latency', color='blue', marker='o')
plt.plot(RC_sizes, inference_latencies1, label='Inference Latency - test1', color='red', marker='x')
plt.plot(RC_sizes, inference_latencies2, label='Inference Latency - test2', color='green', marker='x')
plt.title('Occupancy Detection - ESN Latency - run_with_ip: ' + str(run_with_ip))
plt.xlabel('Reservoir Size')
plt.ylabel('Latency (ms)')
plt.legend()
plt.savefig(f'./{result_folder}/occupancy_detection_esn_latencies.png')