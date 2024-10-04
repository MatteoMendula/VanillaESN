import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from time import time_ns

from utils import calculate_f1_score

training_data_location = 'occupancy_detection/datatraining.txt'
testing_data_location1 = 'occupancy_detection/datatest.txt'
testing_data_location2 = 'occupancy_detection/datatest2.txt'

# read data from txt file datatrainin.txt
training_data = pd.read_csv(training_data_location, sep=',', header=0)
testing_data1 = pd.read_csv(testing_data_location1, sep=',', header=0)
testing_data2 = pd.read_csv(testing_data_location2, sep=',', header=0)

# Calculate the number of 0s and 1s in the labels
num_pos = (training_data['Occupancy'] == 1).sum()
num_neg = (training_data['Occupancy'] == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

# Use nn.BCEWithLogitsLoss and pass the pos_weight argument
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Sample data loading and preparation
# Replace with your dataset loading method (CSV, database, etc.)
# Assuming you have a pandas DataFrame 'df' with the provided columns.

# Example dataset loading (you may already have this in place)
# df = pd.read_csv('path_to_your_data.csv')

# Preprocessing function to create time-windowed data
def create_sequences(data, target, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        label = target[i+window_size]  # Target corresponds to the next step
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Binary classification output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take only the output of the last time step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Hyperparameters
input_size = 5  # We are using 5 features ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
hidden_size = 50
num_layers = 1
num_epochs = 10
learning_rate = 0.001
window_sizes = [1, 3, 5, 10]  # Testing these window sizes

training_latencies1 = []
inference_latencies1 = []
training_latencies2 = []
inference_latencies2 = []

accuracies_test1 = []
f1_scores_test1 = []
accuracies_test2 = []
f1_scores_test2 = []

# Preparing the dataset
def prepare_data(df_train, df_test, window_size):
    features_training = df_train[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
    labels_training = df_train['Occupancy'].values

    features_test = df_test[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
    labels_test = df_test['Occupancy'].values

    # Scaling the features
    scaler = MinMaxScaler()
    features_scaled_training = scaler.fit_transform(features_training)
    features_scaled_test = scaler.transform(features_test)

    # Creating sequences for LSTM
    X_train, y_train = create_sequences(features_scaled_training, labels_training, window_size)
    X_test, y_test = create_sequences(features_scaled_test, labels_test, window_size)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

# Training and evaluation function
def train_and_evaluate(df_train, df_test, window_size, training_latencies, inference_latencies, accuracies, f1_scores):
    print(f"Training with window size: {window_size}")
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df_train, df_test, window_size)

    # Initialize the model
    model = LSTMModel(input_size=5, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    model.train()
    start_time = time_ns()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()  # Remove singleton dimension
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    train_time = (time_ns() - start_time)
    training_latencies.append(train_time)

    # Evaluating the model
    model.eval()
    with torch.no_grad():
        
        start_time = time_ns()
        test_outputs = model(X_test).squeeze()
        test_predictions = (test_outputs > 0.5).float()  # Binarize predictions
        inf_time = (time_ns() - start_time)
        inference_latencies.append(inf_time)

        accuracy = accuracy_score(y_test, test_predictions)
        f1 = calculate_f1_score(y_test, test_predictions)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        print(f'Window Size: {window_size}, Test Accuracy: {accuracy:.4f}')

    # Return actual and predicted values for plotting
    return y_test, test_predictions

# Plotting function
def plot_predictions(y_true, y_pred, window_size, test1 = True):
    plt.figure(figsize=(10, 6))
    
    # Plot the actual values
    plt.plot(y_true, label="Actual", alpha=0.7, color='blue', linestyle='--')
    
    # Plot the predicted values
    plt.plot(y_pred, label="Predicted", alpha=0.7, color='red', linestyle='-')
    
    plt.title(f"Actual vs Predicted Occupancy - Window Size: {window_size}")
    plt.xlabel("Time Steps")
    plt.ylabel("Occupancy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./LSTM_results/{"test1" if test1 else "test2"}/occupancy_detection_lstm_size_{window_size}.png')


if __name__ == '__main__':

    # Running the evaluation for all window sizes and plotting predictions
    for window_size in window_sizes:
        y_true, y_pred = train_and_evaluate(training_data, testing_data1, window_size, training_latencies1, inference_latencies1, accuracies_test1, f1_scores_test1)
        plot_predictions(y_true.numpy(), y_pred.numpy(), window_size)


    # test on the second dataset
    # Running the evaluation for all window sizes and plotting predictions
    for window_size in window_sizes:
        y_true, y_pred = train_and_evaluate(training_data, testing_data2, window_size, training_latencies2, inference_latencies2, accuracies_test2, f1_scores_test2)
        plot_predictions(y_true.numpy(), y_pred.numpy(), window_size, False)

    # Plotting the training and inference latencies
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, training_latencies1, label="Training Latency - dataset 1", marker='o')
    plt.plot(window_sizes, training_latencies2, label="Training Latency - dataset 2", marker='x')
    plt.title("LSTM Training Latencies")
    plt.xlabel("Window Size")
    plt.ylabel("Latency (ns)")
    plt.legend()
    plt.savefig('./LSTM_results/occupancy_detection_lstm_training_latencies.png')

    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, inference_latencies1, label="Inference Latency - dataset 1", marker='o')
    plt.plot(window_sizes, inference_latencies2, label="Inference Latency - dataset 2", marker='x')
    plt.title("LSTM Inference Latencies")
    plt.xlabel("Window Size")
    plt.ylabel("Latency (ns)")
    plt.legend()
    plt.savefig('./LSTM_results/occupancy_detection_lstm_inference_latencies.png')

    # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, accuracies_test1, label="Test 1 Accuracy", marker='o')
    plt.plot(window_sizes, accuracies_test2, label="Test 2 Accuracy", marker='x')
    plt.title("LSTM Occupancy Detection Accuracy")
    plt.xlabel("Window Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('./LSTM_results/occupancy_detection_lstm_accuracies.png')
    
    # Plotting the F1 scores    
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, f1_scores_test1, label="Test 1 F1 Score", marker='o')
    plt.plot(window_sizes, f1_scores_test2, label="Test 2 F1 Score", marker='x')
    plt.title("LSTM Occupancy Detection F1 Score")
    plt.xlabel("Window Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig('./LSTM_results/occupancy_detection_lstm_f1_scores.png')