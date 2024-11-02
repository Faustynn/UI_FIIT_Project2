import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import configparser
import time
import sys
from sklearn.model_selection import train_test_split

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = configparser.ConfigParser()
if len(sys.argv) > 1:
    data_directory = "../data"
    config.read('../config/config.txt')
else:
    data_directory = "data"
    config.read('config/config.txt')

# Barier setup
barriers = {
    0: {'horizontal': -500, 'vertical': 500},  # R
    1: {'horizontal': -500, 'vertical': -500}, # G
    2: {'horizontal': 500, 'vertical': 500},   # B
    3: {'horizontal': 500, 'vertical': -500},  # P
}

# NN def.
class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 4)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.leaky_relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        return self.fc3(out)

# Dataset def.
class PointsDataset(Dataset):
    def __init__(self, points, labels):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]

    def __len__(self):
        return len(self.points)

# loss func.
def barrier_loss(outputs, labels, points):
    criterion = nn.CrossEntropyLoss()
    base_loss = criterion(outputs, labels)

    penalties = torch.zeros_like(labels, dtype=torch.float, device=device)
    for i, label in enumerate(labels):
        label = label.item()
        x, y = points[i]
        horiz, vert = barriers[label]['horizontal'], barriers[label]['vertical']
        if not ((x <= horiz and y <= vert) if label in [0, 2] else (x >= horiz and y >= vert)):
            penalties[i] = 2.0

    return base_loss + penalties.mean()

# Init start points
def init_start_points():
    start_points = {
        0: eval(config['start-bods']['R']),
        1: eval(config['start-bods']['G']),
        2: eval(config['start-bods']['B']),
        3: eval(config['start-bods']['P']),
    }
    points, labels = [], []
    for label, coords in start_points.items():
        for x, y in coords:
            points.append([x, y])
            labels.append(label)
    return np.array(points), np.array(labels)

# Load data
def load_data_from_directories(directory_path):
    points, labels = [], []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    x, y, label = line.strip().split(',')
                    points.append([float(x), float(y)])
                    labels.append(int(label))
    return np.array(points), np.array(labels)

def train_model(model, train_loader, optimizer, early_stopping_patience, test_loader):
    best_accuracy, epochs_without_improvement = 0.0, 0
    model.to(device)
    while epochs_without_improvement < early_stopping_patience:
        model.train()
        running_loss = 0.0
        for batch_points, batch_labels in train_loader:
            batch_points, batch_labels = batch_points.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_points)
            loss = barrier_loss(outputs, batch_labels, batch_points)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for batch_points, batch_labels in test_loader:
                batch_points, batch_labels = batch_points.to(device), batch_labels.to(device)
                outputs = model(batch_points)
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        accuracy = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            if(len(sys.argv) > 1):
                torch.save(model.state_dict(), '../model/model.pth')
            else:
                torch.save(model.state_dict(), 'model/model.pth')
            print(f"Beter model saved with acurancy: {best_accuracy:.2f}")
        else:
            epochs_without_improvement += 1

# Classify points
def classify(model, num_points):
    model.eval()
    generated_points = []
    labels = []

    ranges = {
        0: ((-5000, 500), (-5000, 500)),  # R (0)
        1: ((500, 5000), (-5000, 500)),   # G (1)
        2: ((-5000, 500), (500, 5000)),    # B (2)
        3: ((500, 5000), (500, 5000))      # P (3)
    }

    start_points, start_labels = init_start_points()
    generated_points.extend(start_points)
    labels.extend(start_labels)

    generated_points_set = set(map(tuple, generated_points))
    last_color = -1

    while len(generated_points) < num_points:
        color = np.random.choice([0, 1, 2, 3])  # 0 - R, 1 - G, 2 - B, 3 - P
        while color == last_color:
            color = np.random.choice([0, 1, 2, 3])

        x_range, y_range = ranges[color]

        point = np.random.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]]).astype(np.float32)
        point_tuple = (point[0], point[1])

        if point_tuple not in generated_points_set:
            generated_points.append(point_tuple)
            generated_points_set.add(point_tuple)
            labels.append(color)
            last_color = color

    generated_points = np.array(generated_points)
    points_tensor = torch.tensor(generated_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(points_tensor)
        buff, predicted_labels = torch.max(outputs, 1)

    return generated_points, predicted_labels.cpu().numpy(),labels

def visualize_points(points, labels):
    colors = ['blue', 'purple', 'red', 'green']
    points = np.array(points)
    labels = np.array(labels)

    plt.figure(figsize=(10, 10))

    for label, color in enumerate(colors):
        class_points = points[labels == label]
        plt.scatter(class_points[:, 0], class_points[:, 1], color=color, alpha=0.6, s=8, label=f"Class {label}")

    plt.plot([-5000, 500], [-500, -500], color='#8b0000', linewidth=2)
    plt.plot([500, 500], [-500, 5000], color='#8b0000', linewidth=2)

    plt.plot([-5000, 500], [500, 500], color='#00008b', linewidth=2)
    plt.plot([500, 500], [500, -5000], color='#00008b', linewidth=2)

    plt.plot([-500, 5000], [-500, -500], color='#008b2d', linewidth=2)
    plt.plot([-500, -500], [-500, 5000], color='#008b2d', linewidth=2)

    plt.plot([-500, 5000], [500, 500], color='#ef00fb', linewidth=2)
    plt.plot([-500, -500], [500, -5000], color='#ef00fb', linewidth=2)

    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid()
    plt.show()
def calculate_accuracy(generated_points, predicted_labels,init_labels):
    correct = 0
    for i in range(len(generated_points)):
        if predicted_labels[i] == init_labels[i]:
            correct += 1
    print(f"Accuracy: {correct / len(generated_points) * 100:.2f}%")

# Main
if __name__ == "__main__":
    start_time = time.time()

    if(len(sys.argv) > 1):
        seed=int(sys.argv[1])
    else:
        seed = int(config['generate']['seed'])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    points, labels = init_start_points()
    X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=seed)

    train_dataset = PointsDataset(X_train, y_train)
    test_dataset = PointsDataset(X_test, y_test)

    batch_size = int(config['neuron-settings']['batch-size'])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = ClassifierNN()
    if len(sys.argv) > 1:
        model_path = '../model/model.pth'
    else:
        model_path = 'model/model.pth'

    if os.path.exists(model_path):
        print("Model loaded")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        learning_rate = float(config['neuron-settings']['learning-rate'])
        early_stopping_patience = int(config['neuron-settings']['early-stopping-patience'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, optimizer, early_stopping_patience, test_loader=test_loader)

        torch.save(model.state_dict(), model_path)
        print("Model trained and saved")

    num_points_to_classify = int(config['generate']['num_generate'])
    generated_points, predicted_labels,init_labels = classify(model, num_points_to_classify)
    visualize_points(generated_points, predicted_labels)
    calculate_accuracy(generated_points, predicted_labels,init_labels)

    print(f"Execution time: {time.time() - start_time:.2f} seconds")