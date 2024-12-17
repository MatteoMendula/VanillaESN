import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from wesad_utils import calculate_f1_score, WESAD_EchoStateNetworkWithIP_FedIp
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

BASE_DIR = '/home/matteo/Documents/postDoc/CTTC/WESAD'
subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

n_workers = 3
selected_subjects = subjects[:n_workers]

tester_subject = 10
selected_subjects.append(tester_subject)

binary_task = True

all_subjects_dfs = []
for s in selected_subjects:

    parsed_df_file_location_parent = f'{BASE_DIR}/S{s}'
    file_name = f'S{s}_esn.pkl'
    if file_name in os.listdir(parsed_df_file_location_parent):
        print("file exists")
        df = pd.read_pickle(f'{parsed_df_file_location_parent}/{file_name}')
    else:

        data = pd.read_pickle(f'{BASE_DIR}/S{s}/S{s}.pkl')

        chest_signal = data["signal"]["chest"]

        flattened_features = {}
        for key in ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']:
            if isinstance(chest_signal[key], np.ndarray) and chest_signal[key].ndim > 1:
                for i in range(chest_signal[key].shape[1]):
                    flattened_features[f"{key}_{i}"] = chest_signal[key][:, i]
            else:
                flattened_features[key] = chest_signal[key]

        features = pd.DataFrame(flattened_features)
        target = data["label"]
        # parse target to dataframe
        target = pd.DataFrame(target, columns=['label'])
        subject = pd.DataFrame([s] * len(target), columns=['subject'])

        # Combine features and target
        df = pd.concat([features, target, subject], axis=1)

        if binary_task:
            df = df[(df["label"] == 1) | (df["label"] == 2)]
            df["label"] = df["label"] - 1
        
        df.to_pickle(f'{parsed_df_file_location_parent}/{file_name}')

    all_subjects_dfs.append(df)
    print(df.columns)

# Combine all subjects
all_subjects_df = pd.concat(all_subjects_dfs)
worker_data = []
local_sample_sizes = []

features_test = all_subjects_df[all_subjects_df['subject'] == tester_subject].drop(columns=['label', 'subject']).values
targets_test = all_subjects_df[all_subjects_df['subject'] == tester_subject]['label'].values

scaler = StandardScaler()
features_scaled_test = scaler.fit_transform(features_test)

max_train_size = 10000

for s in selected_subjects[:-1]:
    worker_df = all_subjects_df[all_subjects_df['subject'] == s]
    features_train = worker_df.drop(columns=['label', 'subject']).values
    targets_train = worker_df['label'].values
    
    assert set(targets_train) == {0, 1}, "Labels for binary classification should be 0 or 1"
    
    # Normalize features
    features_scaled_train = scaler.transform(features_train)

    # worker_data.append((features_scaled_train[:max_train_size], targets_train[:max_train_size]))
    # local_sample_sizes.append(len(targets_train[:max_train_size]))

    worker_data.append((features_scaled_train, targets_train))
    local_sample_sizes.append(len(targets_train))

# Hyperparameters
RC_SIZE = 20
N_ROUNDS = 5

# Tracking metrics
accuracy_test = []

# Initialize workers
workers = [WESAD_EchoStateNetworkWithIP_FedIp(input_size=8, reservoir_size=RC_SIZE, output_size=2) for _ in range(n_workers + 1)]
training_workers = workers[:-1]
tester_worker = workers[-1]

global_W_out = None

# Initialize global gain and bias
global_gain = np.zeros(RC_SIZE)
global_bias = np.zeros(RC_SIZE)

# Federated Training Loop
for round_idx in range(1, N_ROUNDS + 1):
    print(f"--- Round {round_idx} ---")

    # Local updates
    local_gain_updates = []
    local_bias_updates = []
    local_W_out_updates = []

    for worker_idx, worker in enumerate(training_workers):
        print(f"Worker {worker_idx + 1} training...")

        pred = worker.predict(features_scaled_test)

        plt.figure(figsize=(10, 6))
        plt.plot(pred, label='predictions_test1')
        plt.plot(targets_test, label='predictions_test1')
        plt.title(f'Round {round_idx} - Worker {worker_idx}')
        plt.legend()
        plt.grid()
        plt.savefig(f'./WESAD_FedIp/{round_idx}_round_{worker_idx}_worker.png')
        acc = accuracy_score(targets_test, pred)
        print(f"Round {round_idx} Worker {worker_idx} Accuracy: {acc:.4f}")

        # print(f"Worker Predictions before round {round_idx}: {np.mean(pred == targets_test)}")
        # print("pred[:10]", pred[:10])
        # plt.figure(figsize=(10, 6))
        # plt.plot(pred, label='Pred')
        # plt.plot(targets_test, label='ground t')
        # plt.show()

        # Set the global gain and bias to each worker
        # add one dimension to global_gain and global_bias
        if global_gain.ndim == 1 and global_bias.ndim == 1:
            global_gain = global_gain.reshape(-1, 1)
            global_bias = global_bias.reshape(-1, 1)
        worker.set_ip_parameters(global_gain, global_bias)

        if global_W_out is not None:
            worker.set_W_out(global_W_out)

        # Train the worker
        features, target = worker_data[worker_idx]

        # worker.fit(features, target)
        worker.fit_online(features, target)
        # worker.fit_rolling_window(features, target)
        # worker.fit_mini_batch(features, target, batch_size=32)

        # pred = worker.predict(features_scaled_test)
        # print(f"Worker Predictions after round {round_idx}: {np.mean(pred == targets_test)}")

        # Collect local gain and bias
        gain, bias = worker.get_ip_parameters()
        local_gain_updates.append(gain)
        local_bias_updates.append(bias)
        local_W_out_updates.append(worker.get_W_out())

    # Aggregate the gain and bias using FedAvg
    global_gain = np.mean(local_gain_updates, axis=0)
    global_bias = np.mean(local_bias_updates, axis=0)
    global_W_out = np.mean(local_W_out_updates, axis=0)
    # global_gain = np.average(local_gain_updates, axis=0, weights=local_sample_sizes)
    # global_bias = np.average(local_bias_updates, axis=0, weights=local_sample_sizes)
    # global_W_out = np.average(local_W_out_updates, axis=0, weights=local_sample_sizes)

    # print("fedavg global_gain", global_gain)
    # print("fedavg global_bias", global_bias)

    tester_worker.set_ip_parameters(global_gain, global_bias)
    tester_worker.check_wout_shape(targets_test)
    tester_worker.set_W_out(global_W_out)

    print("Evaluating on Test ...")
    predictions_test1 = tester_worker.predict(features_scaled_test)  # All workers are identical after aggregation
    acc_test = accuracy_score(targets_test, predictions_test1)
    accuracy_test.append(acc_test)
    print(f"Round {round_idx} Test 1 Accuracy: {acc_test:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_test1, label='predictions_test1')
    plt.plot(targets_test, label='predictions_test1')
    plt.title(f'Round {round_idx} - Worker {worker_idx}')
    plt.legend()
    plt.grid()
    plt.savefig(f'./WESAD_FedIp/{round_idx}_round_aggregator_worker.png')

# Plot Accuracy Over Rounds
plt.figure(figsize=(10, 6))
plt.plot(range(1, N_ROUNDS + 1), accuracy_test, label='Test Accuracy', marker='o')
plt.title('Federated Learning - Accuracy over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('./WESAD_FedIp/federated_learning_accuracy.png')
# plt.show()

