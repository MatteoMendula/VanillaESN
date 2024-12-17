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

from reservoirpy.datasets import narma
from reservoirpy.mat_gen import uniform, bernoulli
from reservoirpy.nodes import IPReservoir
from reservoirpy.nodes import Ridge

import os

def plot_2_series(result_folder, series1, series2, title, xlabel, ylabel, legend1, legend2):

    # create folder if does not exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    plt.figure(figsize=(12, 8))
    plt.plot(series1, label=legend1)
    plt.plot(series2, label=legend2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(f'{result_folder}/{title}.png')

def plot_1_series(result_folder, series_x, series_y, title, xlabel, ylabel, legend):    # create folder if does not exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create folder if does not exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    plt.figure(figsize=(12, 8))
    plt.plot(series_x, series_y, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(f'{result_folder}/{title}.png')

def plot_tsne(result_folder, features, target, title):

    # create folder if does not exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    plt.figure(figsize=(12, 8))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_embedded = tsne.fit_transform(features)
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=target)
    plt.title(title)
    plt.savefig(f'{result_folder}/{title}.png')

def run_experiment(folder, reservoir, readout, run_with_ip, features_train, target_train, features_test1, target_test1, features_test2, target_test2, RC_size, accuracies_test1, f1_scores_test1, accuracies_test2, f1_scores_test2, training_latencies, inference_latencies1, inference_latencies2):
    print(f'Running ESN with reservoir size {RC_size}')
    print("-----------------------------------------")

    target_train = target_train[:, np.newaxis]

    warmup = 100
    start_time = time_ns()
    train_states = reservoir.run(features_train, reset=True)
    readout = readout.fit(train_states, target_train, warmup=warmup)
    train_time = (time_ns() - start_time)
    training_latencies.append(train_time)

    plot_tsne(folder, train_states, target_train, f"RC_size{RC_size} TSNE")

    start_time = time_ns()
    test_states1 = reservoir.run(features_test1)
    y_pred_1 = readout.run(test_states1)
    inf_time = (time_ns() - start_time) 
    inference_latencies1.append(inf_time)
    y_pred_1 = y_pred_1.flatten()
    y_pred_1 = [ 1 if x > 0.5 else 0 for x in y_pred_1]
    plot_2_series(folder, target_test1, y_pred_1, f"RC_size{RC_size} Test1", "Time", "Value", "Target", "Prediction")

    accuracy = accuracy_score(target_test1, y_pred_1)
    f1 = calculate_f1_score(target_test1, y_pred_1)
    print(f"a) RC_size{RC_size} accuracy: {accuracy:.4f} f1: {f1:.4f}")   
    accuracies_test1.append(accuracy)
    f1_scores_test1.append(f1)

    start_time = time_ns()
    test_states2 = reservoir.run(features_test2)
    y_pred_2 = readout.run(test_states2)
    y_pred_2 = y_pred_2.flatten()
    y_pred_2 = [ 1 if x > 0.5 else 0 for x in y_pred_2]
    plot_2_series(folder, target_test2, y_pred_2, f"RC_size{RC_size} Test2", "Time", "Value", "Target", "Prediction")

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

# Parameters (from the paper)
activation = "sigmoid"
units = 50
connectivity = 0.1
sr = 0.95
input_scaling = 0.1
mu = 0.3
warmup = 100
learning_rate = 5e-4
epochs = 100
W_dist = uniform(high=1.0, low=-1.0)
Win_dist = bernoulli

accuracies_test1 = []
f1_scores_test1 = []
accuracies_test2 = []
f1_scores_test2 = []

learning_rates = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
training_latencies = []
inference_latencies1 = []
inference_latencies2 = []

run_with_ip = True
folder = "param_exploration"

for lr in learning_rates:
    reservoir = IPReservoir(
        units,
        sr=sr,
        mu=mu,
        learning_rate=lr,
        input_scaling=input_scaling,
        W=W_dist,
        Win=Win_dist,
        rc_connectivity=connectivity,
        input_connectivity=connectivity,
        activation=activation,
        epochs=epochs,
    )
    readout = Ridge(ridge=1e-7)
    _folder = f"{folder}/learning_rate_{lr}"
    run_experiment(_folder, reservoir, readout, run_with_ip, features_scaled_train, target_train, features_scaled_test1, target_test1, features_scaled_test2, target_test2, units, accuracies_test1, f1_scores_test1, accuracies_test2, f1_scores_test2, training_latencies, inference_latencies1, inference_latencies2)

plot_1_series(folder, learning_rates, training_latencies, "Training Latency", "ip lr", "Latency", "Training Latency")
plot_1_series(folder, learning_rates, inference_latencies1, "Inference Latency Test1", "ip lr", "Latency", "Inference Latency")
plot_1_series(folder, learning_rates, inference_latencies2, "Inference Latency Test2", "ip lr", "Latency", "Inference Latency")
plot_1_series(folder, learning_rates, accuracies_test1, "Accuracy Test1", "ip lr", "Accuracy", "Accuracy")
plot_1_series(folder, learning_rates, accuracies_test2, "Accuracy Test2", "ip lr", "Accuracy", "Accuracy")
plot_1_series(folder, learning_rates, f1_scores_test1, "F1 Test1", "ip lr", "F1", "F1")
plot_1_series(folder, learning_rates, f1_scores_test2, "F1 Test2", "ip lr", "F1", "F1")